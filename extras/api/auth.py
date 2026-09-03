"""
Q-Knee API authentication & role-based access control.

Cryptography & token pipeline:
    - Password hashing/verification via `argon2-cffi` (Argon2id) — the
      OWASP-recommended default for new applications, and avoids the
      well-known `passlib[bcrypt]` incompatibility with `bcrypt>=4.1`
      (passlib's bcrypt backend reads a `bcrypt.__about__` attribute that
      newer `bcrypt` releases removed).
    - JWT access-token signing/verification (`HS256`) via `PyJWT`, with a
      fixed `ACCESS_TOKEN_EXPIRE_MINUTES = 60` lifetime.

User store:
    A single SQLAlchemy 2.0 `UserRepository`, backed by `$DATABASE_URL`
    (PostgreSQL, or any other SQLAlchemy-supported dialect, in production)
    or a local persistent SQLite file (`sqlite:///./qknee_users.db`) when
    `$DATABASE_URL` is unset — the correct, zero-config default for a
    single-node deployment. The `User` table is created automatically at
    import time via `Base.metadata.create_all(engine)`.

    The `User` ORM model lives HERE, in `qknee/api/auth.py`, deliberately
    NOT in `qknee/models/user.py` (even though that's the more obvious
    location): `qknee/models/__init__.py` eagerly imports
    `qknee.models.vqc_data_reuploading`, which imports torch/pennylane at
    module-import time. `qknee/api/server.py` goes to considerable
    documented lengths (see its own module docstring and `get_backend()`)
    to keep torch/pennylane out of the API's cold-start/import path, so a
    free-tier host's boot stays fast and under its memory ceiling. Adding
    `from qknee.models.user import User` here would force the whole
    `qknee.models` package (and therefore torch/pennylane) to import just
    to authenticate a request — defeating that entirely. So the ORM model
    stays a plain SQLAlchemy class in this module instead.

    Every account carries one of three roles (`ROLES` below) — only
    `radiologist` is permitted to run diagnostic inference (`/predict`,
    `/explain`, `/report`); `researcher` and `clinical_auditor` are both
    read-only/non-inference roles (Research Observers). No accounts ship
    pre-seeded: the store starts empty and is populated only via
    `POST /api/v1/auth/register`.

    PRODUCTION CAVEAT: `/register` currently lets a caller self-assign any
    role, including `radiologist` — acceptable for this research-
    prototype/hackathon demo (every diagnostic response elsewhere in this
    codebase already carries a "not for clinical use" disclaimer), but a
    real clinical deployment must gate `radiologist` issuance behind admin
    approval or an invite token rather than open self-service registration.

Route protection:
    `get_current_user` extracts and validates the `Authorization: Bearer
    <token>` header (via `OAuth2PasswordBearer`) and resolves it to the
    live user record (also rejecting a token for a since-deactivated
    account). `require_role([...])` builds on top of it to reject (403)
    any authenticated user whose role isn't in the allowed set — used by
    `qknee.api.server` to guard `/predict`, `/explain`, and `/report`.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional, Sequence

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Boolean, DateTime, String, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

logger = get_logger(__name__)
_config = load_config()

# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #

ROLES: tuple[str, ...] = ("radiologist", "researcher", "clinical_auditor")
DEFAULT_ROLE = "researcher"

# The only role permitted to run diagnostic inference (`/predict`,
# `/explain`, `/report`) — `researcher` and `clinical_auditor` are both
# read-only "Research Observer" tiers. See `qknee.api.server`'s route
# wiring (`require_role(INFERENCE_ROLES)`).
INFERENCE_ROLES: tuple[str, ...] = ("radiologist",)


# --------------------------------------------------------------------------- #
# Pydantic v2 schemas
# --------------------------------------------------------------------------- #

_PASSWORD_SPECIAL_CHARS = re.compile(r"[^a-zA-Z0-9]")

# Deliberately NOT `pydantic.EmailStr`: that type requires the optional
# `email-validator` package to be installed, and importing it (even just to
# declare a field's type — the check happens at class-definition time, when
# Pydantic compiles the model) raises `ImportError: email-validator is not
# installed, run pip install 'pydantic[email]'` if it isn't. `requirements.txt`
# does list `email-validator` as a safeguard, but this module must still
# import cleanly without it — a bare `str` field plus this regex validator
# gives "good enough" structural validation (not full RFC 5322 compliance)
# with zero external dependencies, so a missing/failed `email-validator`
# install on a host like Render can never take the whole API down.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email_format(email: str) -> str:
    email = email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise ValueError("Invalid email address format")
    return email


def _validate_password_strength(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if not any(char.isalpha() for char in password):
        raise ValueError("Password must contain at least one letter.")
    if not any(char.isdigit() for char in password):
        raise ValueError("Password must contain at least one digit.")
    if not _PASSWORD_SPECIAL_CHARS.search(password):
        raise ValueError("Password must contain at least one special character.")
    return password


class UserCreate(BaseModel):
    """`/register` request body."""

    email: str = Field(..., description="Unique institutional email address.")
    password: str = Field(..., max_length=128, description="Plaintext password; never stored or logged.")
    full_name: str = Field(..., min_length=1, max_length=128)
    role: str = Field(
        default=DEFAULT_ROLE,
        description=f"One of {ROLES}; defaults to '{DEFAULT_ROLE}' if omitted.",
    )

    @field_validator("email")
    @classmethod
    def _check_email_format(cls, value: str) -> str:
        return _validate_email_format(value)

    @field_validator("password")
    @classmethod
    def _check_password_strength(cls, value: str) -> str:
        return _validate_password_strength(value)


class UserLogin(BaseModel):
    """`/login` request body. `username` carries the account's email address
    — named `username` (not `email`) so it lines up with the OAuth2
    password-grant convention `oauth2_scheme`'s `tokenUrl` implies."""

    username: str
    password: str

    @field_validator("username")
    @classmethod
    def _check_username_email_format(cls, value: str) -> str:
        return _validate_email_format(value)


class UserResponse(BaseModel):
    """Public-facing user profile — never includes the password hash."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    full_name: str
    role: str
    created_at: datetime
    is_active: bool


class Token(BaseModel):
    """`/login`'s response: a bearer JWT plus the authenticated user's profile."""

    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int = _config.api.access_token_expire_minutes
    user: UserResponse


class TokenData(BaseModel):
    """Decoded JWT payload, validated before being trusted by any dependency."""

    email: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None


# --------------------------------------------------------------------------- #
# Password hashing (Argon2id)
# --------------------------------------------------------------------------- #

_password_hasher = PasswordHasher()


def hash_password(raw_password: str) -> str:
    """Hashes a plaintext password with Argon2id. Returns the encoded hash
    string (algorithm, parameters, salt, and digest all self-contained —
    nothing else needs to be stored alongside it)."""
    return _password_hasher.hash(raw_password)


def verify_password(raw_password: str, hashed_password: str) -> bool:
    """Constant-time-verifies `raw_password` against a stored Argon2 hash.
    Returns `False` (never raises) for a wrong password or a malformed/
    foreign hash string, so callers can treat this as a plain predicate.

    `InvalidHashError` is caught alongside `VerificationError`/
    `VerifyMismatchError` because it does NOT subclass `Argon2Error` the
    way the other two do (it subclasses `ValueError` directly) — argon2-cffi
    raises it for a `hashed_password` string that isn't recognizable Argon2
    output at all (e.g. a stray/corrupted store entry), which should still
    resolve to "verification failed," not an unhandled exception.
    """
    try:
        return _password_hasher.verify(hashed_password, raw_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# --------------------------------------------------------------------------- #
# JWT access tokens
# --------------------------------------------------------------------------- #

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = _config.api.access_token_expire_minutes

# `$QKNEE_JWT_SECRET_KEY` (this project's own name) takes precedence when
# both are set; `$SECRET_KEY` (the generic name a platform like
# Render/Railway/Heroku often auto-populates or that an operator reaches
# for by habit) is accepted as a fallback; `_config.api.jwt_secret_key`
# (config.yaml's dev-only default) is the last resort if neither env var
# is set.
_SECRET_KEY = os.getenv("QKNEE_JWT_SECRET_KEY") or os.getenv("SECRET_KEY") or _config.api.jwt_secret_key
_INSECURE_DEFAULT_SECRET_KEY = "INSECURE-DEV-ONLY-CHANGE-ME-VIA-QKNEE_JWT_SECRET_KEY-ENV-VAR"

if _SECRET_KEY == _INSECURE_DEFAULT_SECRET_KEY:
    logger.warning(
        "qknee.api.auth is signing JWTs with the INSECURE DEFAULT dev secret key. "
        "Set the $QKNEE_JWT_SECRET_KEY (or $SECRET_KEY) environment variable before "
        "deploying anywhere reachable outside a local dev machine — tokens signed with "
        "the default key are forgeable by anyone who has read this source file."
    )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Signs a new `HS256` JWT. `data` is merged as-is into the payload
    (callers pass `{"sub": user.email, "user_id": user.id, "role":
    user.role}`), plus `iat`/`exp` timestamps — `exp` is `expires_delta`
    from now (default `ACCESS_TOKEN_EXPIRE_MINUTES`)."""
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {**data, "iat": now, "exp": expire}
    return jwt.encode(payload, _SECRET_KEY, algorithm=ALGORITHM)


def _credentials_exception(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def decode_access_token(token: str) -> TokenData:
    """Verifies signature + expiry and returns the decoded `TokenData`.
    Raises a 401 `HTTPException` (never a raw `jwt` exception) on any
    failure — expired, malformed, wrong signature, or missing claims."""
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise _credentials_exception("Access token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise _credentials_exception("Could not validate credentials") from exc

    email = payload.get("sub")
    user_id = payload.get("user_id")
    role = payload.get("role")
    if email is None or user_id is None or role is None:
        raise _credentials_exception("Access token is missing required claims")
    return TokenData(email=email, user_id=user_id, role=role)


# --------------------------------------------------------------------------- #
# User store: SQLAlchemy-backed `User` ORM model + `UserRepository`
# --------------------------------------------------------------------------- #

class Base(DeclarativeBase):
    pass


class User(Base):
    """The `User` table. `email` is the account's unique identity — stored
    lowercased so lookups are effectively case-insensitive without a
    separate shadow column."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    def to_response(self) -> UserResponse:
        return UserResponse(
            id=self.id,
            email=self.email,
            full_name=self.full_name,
            role=self.role,
            created_at=self.created_at,
            is_active=self.is_active,
        )


_SQLITE_FALLBACK_URL = "sqlite:///./qknee_users.db"
_SQLITE_MEMORY_URL = "sqlite:///:memory:"
_RAW_DATABASE_URL = os.getenv("DATABASE_URL") or _SQLITE_FALLBACK_URL

# Render/Heroku-provisioned Postgres add-ons commonly hand back the legacy
# `postgres://` scheme, which SQLAlchemy 2.0 rejects outright
# (`NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:postgres`) —
# normalize it to `postgresql://` before it ever reaches `create_engine`.
if _RAW_DATABASE_URL.startswith("postgres://"):
    _RAW_DATABASE_URL = "postgresql://" + _RAW_DATABASE_URL[len("postgres://"):]


def _build_engine(database_url: str):
    """Builds the engine and eagerly creates the `users` table, so a bad
    connection string or an unreachable host fails HERE — inside this
    function, at import time — rather than as a hard-to-diagnose 500 on the
    first real request. Callers (module scope, below) catch any failure and
    fall back further down the chain instead of letting it kill the whole
    process: a misconfigured/unreachable `$DATABASE_URL` (SSL handshake
    latency, DNS failure, wrong credentials, ...) on a platform like Render
    must never take the entire API down before Uvicorn can even bind a
    port.

    `sqlite:///:memory:` needs `StaticPool` (a single, never-closed
    connection shared by every checkout) — SQLAlchemy's default pooling
    otherwise hands each connection its own private in-memory database, so
    a request served on a different pooled connection than the one that
    created the `users` table would see it as empty/missing.
    """
    is_sqlite = database_url.startswith("sqlite")
    is_memory = database_url == _SQLITE_MEMORY_URL
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False} if is_sqlite else {},
        poolclass=StaticPool if is_memory else None,
        pool_pre_ping=not is_sqlite,
        future=True,
    )
    Base.metadata.create_all(engine)
    return engine


try:
    _engine = _build_engine(_RAW_DATABASE_URL)
    DATABASE_URL = _RAW_DATABASE_URL
except Exception as exc:  # noqa: BLE001 - any driver/connectivity/syntax failure degrades, never crashes
    logger.error(
        "Neon DB connection failed (%s); falling back to the local SQLite store at %s. Set a "
        "valid $DATABASE_URL (postgresql://...) to use a real database.",
        exc, _SQLITE_FALLBACK_URL,
    )
    try:
        _engine = _build_engine(_SQLITE_FALLBACK_URL)
        DATABASE_URL = _SQLITE_FALLBACK_URL
    except Exception as exc2:  # noqa: BLE001 - e.g. a read-only container filesystem
        logger.error(
            "Local SQLite file store at %s is also unavailable (%s); falling back to an "
            "in-memory SQLite database so the API can still boot. User accounts will NOT "
            "persist across restarts until a working $DATABASE_URL is configured.",
            _SQLITE_FALLBACK_URL, exc2,
        )
        DATABASE_URL = _SQLITE_MEMORY_URL
        _engine = _build_engine(DATABASE_URL)

_SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)


class UserAlreadyExistsError(Exception):
    """Raised by `UserRepository.create_user` when the email is already
    registered."""


class UserRepository:
    """SQLAlchemy-backed user repository. Opens and closes a short-lived
    `Session` per call (rather than holding one long-lived session) so
    concurrent requests running on Starlette's sync threadpool never share
    a `Session` object across threads."""

    def __init__(self, session_factory: sessionmaker = _SessionLocal) -> None:
        self._session_factory = session_factory

    def get_by_email(self, email: str) -> Optional[User]:
        with self._session_factory() as session:  # type: Session
            return session.execute(
                select(User).where(User.email == email.strip().lower())
            ).scalar_one_or_none()

    def create_user(self, user_create: UserCreate) -> User:
        role = user_create.role or DEFAULT_ROLE
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")

        user = User(
            email=user_create.email.strip().lower(),
            hashed_password=hash_password(user_create.password),
            full_name=user_create.full_name,
            role=role,
        )
        with self._session_factory() as session:  # type: Session
            session.add(user)
            try:
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise UserAlreadyExistsError(f"An account already exists for '{user_create.email}'") from exc
            session.refresh(user)
        return user

    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Returns the `User` if `email`/`password` are valid and the
        account is active, else `None`."""
        user = self.get_by_email(email)
        if user is None:
            # Returns immediately rather than hashing a dummy password to
            # equalize timing against the "wrong password" branch below —
            # email-enumeration-via-timing isn't this demo API's threat
            # model priority; noted explicitly rather than silently
            # accepted.
            return None
        if not user.is_active:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user


user_store = UserRepository()


# --------------------------------------------------------------------------- #
# FastAPI dependencies: authentication + RBAC
# --------------------------------------------------------------------------- #

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=True)


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """Resolves the `Authorization: Bearer <token>` header to a live user
    profile. Raises 401 if the token is missing/invalid/expired, if it
    decodes fine but no longer names an existing account (e.g. deleted
    after the token was issued), or if the account has since been
    deactivated."""
    token_data = decode_access_token(token)
    user = user_store.get_by_email(token_data.email)
    if user is None:
        raise _credentials_exception("User for this access token no longer exists")
    if not user.is_active:
        raise _credentials_exception("This account has been deactivated")
    return user.to_response()


def require_role(required_roles: Sequence[str]) -> Callable[[UserResponse], UserResponse]:
    """Builds a FastAPI dependency that requires the authenticated user's
    role to be one of `required_roles`, e.g.:

        @app.post("/predict")
        def predict(..., user: UserResponse = Depends(require_role(["radiologist"]))):
            ...

    Layers on top of `get_current_user`, so an unauthenticated request
    still gets 401 (not 403) — 403 is reserved for "authenticated, but
    wrong role."
    """
    allowed = set(required_roles)

    def _dependency(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
        if current_user.role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{current_user.role}' is not permitted to access this resource "
                    f"(requires one of: {sorted(allowed)})."
                ),
            )
        return current_user

    return _dependency


# --------------------------------------------------------------------------- #
# FastAPI router: /api/v1/auth/{register,login,me}
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_create: UserCreate) -> UserResponse:
    """Registers a new user: hashes the password (Argon2id) and stores the
    account with the requested role (defaulting to `researcher` if
    omitted). 409s if the email is already registered."""
    try:
        stored = user_store.create_user(user_create)
    except UserAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        # 422 (Unprocessable Content/Entity — the constant name changed
        # across starlette versions): the request was well-formed JSON but
        # `role` isn't one of `ROLES`.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return stored.to_response()


@router.post("/login", response_model=Token)
def login(credentials: UserLogin) -> Token:
    """Authenticates `username` (email)/`password` and returns a signed JWT
    bearer token (expiring after `ACCESS_TOKEN_EXPIRE_MINUTES`) alongside
    the user's profile metadata."""
    user = user_store.authenticate(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email, "user_id": user.id, "role": user.role})
    return Token(access_token=access_token, user=user.to_response())


@router.get("/me", response_model=UserResponse)
def me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Returns the authenticated caller's own profile — proves the bearer
    token round-trips correctly and is the simplest possible protected
    endpoint to smoke-test a client's auth integration against."""
    return current_user
