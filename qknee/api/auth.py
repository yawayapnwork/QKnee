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
    Storage-backend-agnostic via the `UserRepository` interface below, with
    two implementations selected automatically from `$DATABASE_URL` (see
    `qknee.config.loader.StorageConfig`):
        - `SQLAlchemyUserRepository` — used when `$DATABASE_URL` names a
          reachable database (PostgreSQL in production; any other
          SQLAlchemy-supported URL, including a local `sqlite:///...`
          file, also works).
        - `LocalFileUserRepository` — a JSON-file-backed store
          (`qknee/artifacts/users.json` by default, created on first use),
          used whenever `$DATABASE_URL` is unset/empty, OR as an automatic
          fallback if a configured `$DATABASE_URL` fails to connect at
          startup. This is the correct (and default) backend for a
          single-node free-tier deployment (Render, Streamlit Cloud,
          Vercel) with no managed Postgres attached.
    Every account carries one of three roles (`ROLES` below) —
    `radiologist` and `triage_nurse` are the clinical roles permitted to
    run diagnostic inference (`/predict`, `/explain`); `guest_demo` is a
    read-only demo/judge account. No accounts ship pre-seeded: the store
    starts empty and is populated only via `POST /api/v1/auth/signup`.

    PRODUCTION CAVEAT: `/signup` currently lets a caller self-assign any
    role, including the clinical ones — acceptable for this research-
    prototype/hackathon demo (every diagnostic response elsewhere in this
    codebase already carries a "not for clinical use" disclaimer), but a
    real clinical deployment must gate `radiologist`/`triage_nurse`
    issuance behind admin approval or an invite token rather than open
    self-service signup.

Route protection:
    `get_current_user` extracts and validates the `Authorization: Bearer
    <token>` header (via `OAuth2PasswordBearer`) and resolves it to the
    live user record. `require_role([...])` builds on top of it to reject
    (403) any authenticated user whose role isn't in the allowed set —
    used by `qknee.api.server` to guard `/predict` and `/explain`.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, ConfigDict, Field

from qknee.config.loader import load_config, redact_connection_string
from qknee.config.logging_config import get_logger

logger = get_logger(__name__)
_config = load_config()

# --------------------------------------------------------------------------- #
# Roles
# --------------------------------------------------------------------------- #

ROLES: tuple[str, ...] = ("radiologist", "triage_nurse", "guest_demo")
DEFAULT_ROLE = "guest_demo"

# The two clinical roles permitted to run diagnostic inference
# (`/predict`, `/explain`); `guest_demo` is intentionally excluded —
# see `qknee.api.server`'s route wiring.
INFERENCE_ROLES: tuple[str, ...] = ("radiologist", "triage_nurse")


# --------------------------------------------------------------------------- #
# Pydantic v2 schemas
# --------------------------------------------------------------------------- #

class UserCreate(BaseModel):
    """Signup request body."""

    username: str = Field(
        ..., min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$",
        description="Unique, case-insensitive username (letters, digits, '_', '.', '-').",
    )
    password: str = Field(..., min_length=8, max_length=128, description="Plaintext password; never stored or logged.")
    role: str = Field(
        default=DEFAULT_ROLE,
        description=f"One of {ROLES}; defaults to '{DEFAULT_ROLE}' if omitted.",
    )


class UserLogin(BaseModel):
    """Login request body."""

    username: str
    password: str


class UserResponse(BaseModel):
    """Public-facing user profile — never includes the password hash."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    role: str
    created_at: datetime


class Token(BaseModel):
    """`/login`'s response: a bearer JWT plus the authenticated user's profile."""

    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int = _config.api.access_token_expire_minutes
    user: UserResponse


class TokenData(BaseModel):
    """Decoded JWT payload, validated before being trusted by any dependency."""

    username: Optional[str] = None
    role: Optional[str] = None


# --------------------------------------------------------------------------- #
# Password hashing (Argon2id)
# --------------------------------------------------------------------------- #

_password_hasher = PasswordHasher()


def hash_password(plain_password: str) -> str:
    """Hashes a plaintext password with Argon2id. Returns the encoded hash
    string (algorithm, parameters, salt, and digest all self-contained —
    nothing else needs to be stored alongside it)."""
    return _password_hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time-verifies `plain_password` against a stored Argon2 hash.
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
        return _password_hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


# --------------------------------------------------------------------------- #
# JWT access tokens
# --------------------------------------------------------------------------- #

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = _config.api.access_token_expire_minutes

# Checked directly against the environment (not just through
# `_config.api.jwt_secret_key`'s own `$QKNEE_JWT_SECRET_KEY` override —
# see `qknee.config.loader._ENV_OVERRIDES`) so both naming conventions
# work interchangeably: `$QKNEE_JWT_SECRET_KEY` (this project's own name)
# takes precedence when both are set, `$SECRET_KEY` (the generic name a
# platform like Render/Railway/Heroku often auto-populates or that an
# operator reaches for by habit) is accepted as a fallback, and
# `_config.api.jwt_secret_key` (config.yaml's dev-only default) is the
# last resort if neither env var is set.
_SECRET_KEY = os.getenv("QKNEE_JWT_SECRET_KEY") or os.getenv("SECRET_KEY") or _config.api.jwt_secret_key
_INSECURE_DEFAULT_SECRET_KEY = "INSECURE-DEV-ONLY-CHANGE-ME-VIA-QKNEE_JWT_SECRET_KEY-ENV-VAR"

if _SECRET_KEY == _INSECURE_DEFAULT_SECRET_KEY:
    logger.warning(
        "qknee.api.auth is signing JWTs with the INSECURE DEFAULT dev secret key. "
        "Set the $QKNEE_JWT_SECRET_KEY (or $SECRET_KEY) environment variable before "
        "deploying anywhere reachable outside a local dev machine — tokens signed with "
        "the default key are forgeable by anyone who has read this source file."
    )


def create_access_token(username: str, role: str, expires_delta: Optional[timedelta] = None) -> str:
    """Signs a new `HS256` JWT for `username`/`role`, expiring after
    `expires_delta` (default `ACCESS_TOKEN_EXPIRE_MINUTES`)."""
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    payload = {"sub": username, "role": role, "iat": now, "exp": expire}
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

    username = payload.get("sub")
    role = payload.get("role")
    if username is None or role is None:
        raise _credentials_exception("Access token is missing required claims")
    return TokenData(username=username, role=role)


# --------------------------------------------------------------------------- #
# User store: abstract repository + two backends
# (LocalFileUserRepository / SQLAlchemyUserRepository), selected by
# `_build_user_repository()` from `$DATABASE_URL` at the bottom of this
# section.
# --------------------------------------------------------------------------- #

USERS_STORE_PATH = _config.storage.local_users_path
DEFAULT_LOCAL_USERS_PATH = USERS_STORE_PATH  # explicit alias matching this section's new naming


class UserAlreadyExistsError(Exception):
    """Raised by a `UserRepository.create_user` implementation when the
    username is already taken."""


@dataclass
class StoredUser:
    id: str
    username: str
    hashed_password: str
    role: str
    created_at: str  # ISO 8601, UTC

    def to_response(self) -> UserResponse:
        return UserResponse(id=self.id, username=self.username, role=self.role, created_at=self.created_at)


class UserRepository(ABC):
    """Storage-backend-agnostic user repository — `qknee.api.auth`'s
    `/signup`/`/login`/`/me` routes and FastAPI dependencies talk to this
    interface exclusively (via the module-level `user_store` instance
    below), never to a concrete backend directly, so password hashing
    (Argon2id), JWT issuance, and role validation are identical regardless
    of which implementation is active.
    """

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[StoredUser]:
        """Case-insensitive username lookup. Returns `None` if no such
        account exists."""

    @abstractmethod
    def create_user(self, user_create: UserCreate) -> StoredUser:
        """Hashes the password and persists a new account. Raises
        `UserAlreadyExistsError` if the (case-insensitive) username is
        already taken, or `ValueError` if `user_create.role` isn't one of
        `ROLES`."""

    def authenticate(self, username: str, password: str) -> Optional[StoredUser]:
        """Returns the `StoredUser` if `username`/`password` are valid,
        else `None`. Backend-agnostic: implemented once here in terms of
        `get_by_username` + `verify_password`, so every backend gets
        identical authentication semantics for free."""
        user = self.get_by_username(username)
        if user is None:
            # Returns immediately rather than hashing a dummy password to
            # equalize timing against the "wrong password" branch below —
            # username-enumeration-via-timing isn't this demo API's threat
            # model priority; noted explicitly rather than silently
            # accepted.
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user


class LocalFileUserRepository(UserRepository):
    """JSON-file-backed user repository, guarded by a `threading.Lock` for
    read-modify-write safety under FastAPI's threadpool-executed sync
    endpoints. The default backend whenever `$DATABASE_URL` is unset/empty
    — the correct (and zero-config) choice for a single-node free-tier
    deployment (Render, Streamlit Cloud, Vercel) with no managed Postgres
    attached — and also the automatic fallback if a configured
    `$DATABASE_URL` fails to connect at startup (see
    `_build_user_repository`). Not a substitute for a real RDBMS under
    concurrent multi-process deployment — see `SQLAlchemyUserRepository`
    below for that case.
    """

    def __init__(self, path: Path = DEFAULT_LOCAL_USERS_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({"users": {}})

    def _read(self) -> Dict[str, Any]:
        with self._path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: Dict[str, Any]) -> None:
        # Write-to-temp-then-replace so a crash mid-write can never leave
        # users.json truncated/corrupted.
        tmp_path = self._path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        tmp_path.replace(self._path)

    def get_by_username(self, username: str) -> Optional[StoredUser]:
        record = self._read()["users"].get(username.lower())
        return StoredUser(**record) if record is not None else None

    def create_user(self, user_create: UserCreate) -> StoredUser:
        role = user_create.role or DEFAULT_ROLE
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")

        key = user_create.username.lower()
        with self._lock:
            data = self._read()
            if key in data["users"]:
                raise UserAlreadyExistsError(f"Username '{user_create.username}' is already taken")

            stored = StoredUser(
                id=uuid.uuid4().hex,
                username=user_create.username,
                hashed_password=hash_password(user_create.password),
                role=role,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            data["users"][key] = stored.__dict__
            self._write(data)
        return stored


# Backward-compatible alias: `UserStore` was this class's name before the
# `UserRepository` interface was introduced. Existing call sites/tests
# (`UserStore(path)`, `isinstance(x, UserStore)`) keep working unchanged —
# this is the exact same class, not a wrapper.
UserStore = LocalFileUserRepository


def _to_sync_sqlalchemy_url(database_url: str) -> str:
    """Strips a `+<async_driver>` DBAPI qualifier (e.g. `postgresql+asyncpg://`
    -> `postgresql://`) from `database_url`'s dialect. `qknee.api.auth`'s
    routes are defined as sync `def`s (Starlette already runs them in a
    threadpool), so `SQLAlchemyUserRepository` always uses a synchronous
    `Engine` — an async-only driver name like `asyncpg` in an example
    `$DATABASE_URL` is accepted as input but never actually asked to open
    an async connection; SQLAlchemy falls back to whatever *synchronous*
    driver is installed for the same base dialect (`psycopg`/`psycopg2`
    for `postgresql://`, the stdlib `sqlite3` module for `sqlite://`,
    no extra install needed).
    """
    scheme, sep, rest = database_url.partition("://")
    if not sep:
        return database_url
    dialect = scheme.split("+", 1)[0]
    return f"{dialect}://{rest}"


class SQLAlchemyUserRepository(UserRepository):
    """SQLAlchemy-backed user repository — used whenever `$DATABASE_URL`
    names a reachable database. Works against PostgreSQL in production,
    or any other SQLAlchemy-supported synchronous dialect, including a
    local `sqlite:///./users.db` file for a single-node deployment that
    still wants a real SQL store instead of the JSON fallback.

    Construction fails loudly (raises) on any connectivity/driver problem
    — a missing DBAPI driver, an unreachable host, bad credentials — so
    `_build_user_repository()` can catch that here, at startup, and
    degrade to `LocalFileUserRepository` instead of a configured-but-
    broken `$DATABASE_URL` surfacing as a 500 on the first `/signup` or
    `/login` request.
    """

    def __init__(self, database_url: str) -> None:
        from sqlalchemy import Column, MetaData, String, Table, create_engine, select
        from sqlalchemy.exc import IntegrityError

        self._select = select
        self._IntegrityError = IntegrityError

        sync_url = _to_sync_sqlalchemy_url(database_url)
        self._engine = create_engine(sync_url, pool_pre_ping=True, future=True)

        self._metadata = MetaData()
        self._users = Table(
            "qknee_users",
            self._metadata,
            Column("id", String(64), primary_key=True),
            Column("username", String(64), nullable=False),
            Column("username_lower", String(64), nullable=False, unique=True, index=True),
            Column("hashed_password", String(256), nullable=False),
            Column("role", String(32), nullable=False),
            Column("created_at", String(64), nullable=False),
        )
        # Connects and creates the table (if missing) right away, so a
        # bad URL/unreachable host fails here — inside this constructor —
        # rather than lazily on the first real request.
        self._metadata.create_all(self._engine)

    @staticmethod
    def _row_to_user(row: Any) -> StoredUser:
        return StoredUser(
            id=row.id,
            username=row.username,
            hashed_password=row.hashed_password,
            role=row.role,
            created_at=row.created_at,
        )

    def get_by_username(self, username: str) -> Optional[StoredUser]:
        with self._engine.connect() as conn:
            row = conn.execute(
                self._select(self._users).where(self._users.c.username_lower == username.lower())
            ).first()
        return self._row_to_user(row) if row is not None else None

    def create_user(self, user_create: UserCreate) -> StoredUser:
        role = user_create.role or DEFAULT_ROLE
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")

        stored = StoredUser(
            id=uuid.uuid4().hex,
            username=user_create.username,
            hashed_password=hash_password(user_create.password),
            role=role,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    self._users.insert().values(
                        id=stored.id,
                        username=stored.username,
                        username_lower=stored.username.lower(),
                        hashed_password=stored.hashed_password,
                        role=stored.role,
                        created_at=stored.created_at,
                    )
                )
        except self._IntegrityError as exc:
            raise UserAlreadyExistsError(f"Username '{user_create.username}' is already taken") from exc
        return stored


def _build_user_repository() -> UserRepository:
    """Selects the user-repository backend from `$DATABASE_URL`
    (`qknee.config.loader.StorageConfig.database_url`):
        - unset/empty -> `LocalFileUserRepository` (no error, no warning —
          this is the expected, fully-supported configuration for a
          single-node free-tier deployment).
        - set but unreachable/misconfigured -> logs a warning and falls
          back to `LocalFileUserRepository` rather than crashing the API
          at import time.
        - set and reachable -> `SQLAlchemyUserRepository`.
    """
    database_url = _config.storage.database_url.strip()
    if database_url:
        redacted = redact_connection_string(database_url)
        try:
            repository: UserRepository = SQLAlchemyUserRepository(database_url)
            logger.info("User store backend: SQLAlchemyUserRepository (%s)", redacted)
            return repository
        except Exception as exc:  # noqa: BLE001 - any driver/connectivity failure degrades, never crashes
            logger.warning(
                "Failed to initialize SQLAlchemyUserRepository for DATABASE_URL=%s (%s); "
                "falling back to LocalFileUserRepository.", redacted, exc,
            )

    repository = LocalFileUserRepository(DEFAULT_LOCAL_USERS_PATH)
    logger.info("User store backend: LocalFileUserRepository (%s)", DEFAULT_LOCAL_USERS_PATH)
    return repository


user_store: UserRepository = _build_user_repository()


# --------------------------------------------------------------------------- #
# FastAPI dependencies: authentication + RBAC
# --------------------------------------------------------------------------- #

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=True)


def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """Resolves the `Authorization: Bearer <token>` header to a live user
    profile. Raises 401 if the token is missing/invalid/expired, or if it
    decodes fine but no longer names an existing account (e.g. deleted
    after the token was issued)."""
    token_data = decode_access_token(token)
    user = user_store.get_by_username(token_data.username)
    if user is None:
        raise _credentials_exception("User for this access token no longer exists")
    return user.to_response()


def require_role(allowed_roles: Sequence[str]) -> Callable[[UserResponse], UserResponse]:
    """Builds a FastAPI dependency that requires the authenticated user's
    role to be one of `allowed_roles`, e.g.:

        @app.post("/predict")
        def predict(..., user: UserResponse = Depends(require_role(["radiologist"]))):
            ...

    Layers on top of `get_current_user`, so an unauthenticated request
    still gets 401 (not 403) — 403 is reserved for "authenticated, but
    wrong role."
    """
    allowed = set(allowed_roles)

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
# FastAPI router: /api/v1/auth/{signup,login,me}
# --------------------------------------------------------------------------- #

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


@router.post("/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def signup(user_create: UserCreate) -> UserResponse:
    """Registers a new user: hashes the password (Argon2id) and stores the
    account with the requested role (defaulting to `guest_demo` if
    omitted). 409s if the username is already taken."""
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
    """Authenticates `username`/`password` and returns a signed JWT bearer
    token (expiring after `ACCESS_TOKEN_EXPIRE_MINUTES`) alongside the
    user's profile metadata."""
    user = user_store.authenticate(credentials.username, credentials.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(username=user.username, role=user.role)
    return Token(access_token=access_token, user=user.to_response())


@router.get("/me", response_model=UserResponse)
def me(current_user: UserResponse = Depends(get_current_user)) -> UserResponse:
    """Returns the authenticated caller's own profile — proves the bearer
    token round-trips correctly and is the simplest possible protected
    endpoint to smoke-test a client's auth integration against."""
    return current_user
