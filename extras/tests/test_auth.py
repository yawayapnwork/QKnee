"""
Tests for `qknee.api.auth` — password hashing, JWT signing/verification,
the SQLAlchemy-backed `UserRepository`, and the `get_current_user`/
`require_role` FastAPI dependencies — plus the wired-up `/api/v1/auth/*`
endpoints and RBAC guards on `/predict`/`/explain` in `qknee.api.server`.

Every test gets an isolated `UserRepository` (a fresh in-memory SQLite
engine, monkeypatched over the shared `qknee.api.auth.user_store`
singleton) so tests never share or leak state through the real
`qknee_users.db` file.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("fastapi")

import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import qknee.api.auth as auth_module
from qknee.api.auth import (
    DEFAULT_ROLE,
    INFERENCE_ROLES,
    ROLES,
    Base,
    Token,
    TokenData,
    User,
    UserAlreadyExistsError,
    UserCreate,
    UserLogin,
    UserRepository,
    UserResponse,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

pytestmark = [pytest.mark.slow]


def _make_isolated_repository() -> UserRepository:
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    return UserRepository(session_factory=session_factory)


@pytest.fixture
def isolated_store(monkeypatch: pytest.MonkeyPatch) -> UserRepository:
    """A fresh in-memory-SQLite-backed `UserRepository`, also swapped in
    for the module-level singleton so dependencies that reference
    `auth_module.user_store` directly (`get_current_user`) see it too."""
    store = _make_isolated_repository()
    monkeypatch.setattr(auth_module, "user_store", store)
    return store


def _make_user_create(email: str = "alice@hospital.org", password: str = "hunter2R0cks!", **kwargs) -> UserCreate:
    kwargs.setdefault("full_name", "Alice Smith")
    return UserCreate(email=email, password=password, **kwargs)


# --------------------------------------------------------------------------- #
# 1. Password hashing (Argon2id)
# --------------------------------------------------------------------------- #

class TestPasswordHashing:
    def test_hash_password_does_not_return_the_plaintext(self):
        hashed = hash_password("correct horse battery staple!1")
        assert hashed != "correct horse battery staple!1"
        assert hashed.startswith("$argon2id$")

    def test_verify_password_accepts_the_correct_password(self):
        hashed = hash_password("s3cr3t-passw0rd!")
        assert verify_password("s3cr3t-passw0rd!", hashed) is True

    def test_verify_password_rejects_the_wrong_password(self):
        hashed = hash_password("s3cr3t-passw0rd!")
        assert verify_password("wrong-password!1", hashed) is False

    def test_verify_password_rejects_a_malformed_hash_without_raising(self):
        assert verify_password("anything", "not-a-real-argon2-hash") is False

    def test_hash_password_is_salted_and_nondeterministic(self):
        hash_a = hash_password("same-password!1")
        hash_b = hash_password("same-password!1")
        assert hash_a != hash_b  # different random salts
        assert verify_password("same-password!1", hash_a)
        assert verify_password("same-password!1", hash_b)


# --------------------------------------------------------------------------- #
# 2. JWT access tokens
# --------------------------------------------------------------------------- #

class TestJWTAccessTokens:
    def _data(self, email="dr.house@hospital.org", user_id="user-1", role="radiologist"):
        return {"sub": email, "user_id": user_id, "role": role}

    def test_create_and_decode_round_trips_email_userid_and_role(self):
        token = create_access_token(data=self._data())
        token_data = decode_access_token(token)

        assert isinstance(token_data, TokenData)
        assert token_data.email == "dr.house@hospital.org"
        assert token_data.user_id == "user-1"
        assert token_data.role == "radiologist"

    def test_token_is_signed_hs256(self):
        token = create_access_token(data=self._data())
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "HS256"

    def test_default_expiry_is_sixty_minutes(self):
        assert auth_module.ACCESS_TOKEN_EXPIRE_MINUTES == 60
        token = create_access_token(data=self._data())
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["exp"] - payload["iat"] == 60 * 60

    def test_expired_token_is_rejected(self):
        token = create_access_token(data=self._data(), expires_delta=timedelta(seconds=-1))
        with pytest.raises(Exception) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_tampered_token_is_rejected(self):
        token = create_access_token(data=self._data())
        header, payload, signature = token.split(".")
        # Flip a character in the middle of the payload segment (not the
        # last character of the whole token, which can land on an unused
        # base64url padding bit and decode back to the same byte) so the
        # signature verification is guaranteed to fail.
        mid = len(payload) // 2
        flipped_char = "A" if payload[mid] != "A" else "B"
        tampered_payload = payload[:mid] + flipped_char + payload[mid + 1 :]
        tampered = ".".join([header, tampered_payload, signature])

        with pytest.raises(Exception) as exc_info:
            decode_access_token(tampered)
        assert exc_info.value.status_code == 401

    def test_token_signed_with_a_different_secret_is_rejected(self):
        forged = jwt.encode(
            {"sub": "dr.house@hospital.org", "user_id": "user-1", "role": "radiologist"},
            "some-other-secret-key-of-sufficient-length",
            algorithm="HS256",
        )
        with pytest.raises(Exception) as exc_info:
            decode_access_token(forged)
        assert exc_info.value.status_code == 401

    def test_token_missing_required_claims_is_rejected(self):
        malformed = jwt.encode({"sub": "dr.house@hospital.org"}, auth_module._SECRET_KEY, algorithm="HS256")  # no "role"/"user_id"
        with pytest.raises(Exception) as exc_info:
            decode_access_token(malformed)
        assert exc_info.value.status_code == 401


# --------------------------------------------------------------------------- #
# 3. UserRepository
# --------------------------------------------------------------------------- #

class TestUserRepository:
    def test_create_user_hashes_the_password(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(role="radiologist"))
        assert isinstance(stored, User)
        assert stored.hashed_password != "hunter2R0cks!"
        assert verify_password("hunter2R0cks!", stored.hashed_password)

    def test_create_user_defaults_to_researcher_role(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(email="judge@hospital.org"))
        assert stored.role == DEFAULT_ROLE == "researcher"

    def test_create_user_rejects_duplicate_email_case_insensitively(self, isolated_store: UserRepository):
        isolated_store.create_user(_make_user_create(email="Alice@Hospital.org", role="radiologist"))
        with pytest.raises(UserAlreadyExistsError):
            isolated_store.create_user(_make_user_create(email="alice@hospital.org", password="different-pw!1", role="radiologist"))

    def test_create_user_rejects_a_role_outside_the_allowed_set(self, isolated_store: UserRepository):
        bad_user = UserCreate.model_construct(email="mallory@hospital.org", password="password123!", full_name="Mallory", role="admin")
        with pytest.raises(ValueError, match="role must be one of"):
            isolated_store.create_user(bad_user)

    def test_get_by_email_is_case_insensitive(self, isolated_store: UserRepository):
        isolated_store.create_user(_make_user_create(email="Bob@Hospital.org", role="researcher"))
        assert isolated_store.get_by_email("bob@hospital.org") is not None
        assert isolated_store.get_by_email("BOB@HOSPITAL.ORG") is not None

    def test_get_by_email_returns_none_for_unknown_user(self, isolated_store: UserRepository):
        assert isolated_store.get_by_email("nobody@hospital.org") is None

    def test_authenticate_succeeds_with_correct_credentials(self, isolated_store: UserRepository):
        isolated_store.create_user(_make_user_create(email="carol@hospital.org", password="correct-pw1!", role="researcher"))
        user = isolated_store.authenticate("carol@hospital.org", "correct-pw1!")
        assert user is not None
        assert user.email == "carol@hospital.org"

    def test_authenticate_fails_with_wrong_password(self, isolated_store: UserRepository):
        isolated_store.create_user(_make_user_create(email="carol@hospital.org", password="correct-pw1!", role="researcher"))
        assert isolated_store.authenticate("carol@hospital.org", "wrong-pw1!") is None

    def test_authenticate_fails_for_unknown_user(self, isolated_store: UserRepository):
        assert isolated_store.authenticate("ghost@hospital.org", "anything!1") is None

    def test_authenticate_fails_for_deactivated_user(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(email="dana@hospital.org", password="password123!", role="radiologist"))
        with isolated_store._session_factory() as session:
            db_user = session.get(User, stored.id)
            db_user.is_active = False
            session.commit()
        assert isolated_store.authenticate("dana@hospital.org", "password123!") is None

    def test_store_persists_across_repository_instances_via_the_same_engine(self):
        engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True,
    )
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

        UserRepository(session_factory=session_factory).create_user(
            _make_user_create(email="dana@hospital.org", role="radiologist")
        )
        reopened = UserRepository(session_factory=session_factory)
        assert reopened.get_by_email("dana@hospital.org") is not None

    def test_stored_user_to_response_omits_the_password_hash(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(email="erin@hospital.org", role="researcher"))
        response = stored.to_response()
        assert isinstance(response, UserResponse)
        assert not hasattr(response, "hashed_password")


# --------------------------------------------------------------------------- #
# 4. Pydantic v2 schemas
# --------------------------------------------------------------------------- #

class TestSchemas:
    def test_roles_tuple_matches_the_required_default_roles(self):
        assert set(ROLES) == {"radiologist", "researcher", "clinical_auditor"}

    def test_inference_roles_is_radiologist_only(self):
        assert set(INFERENCE_ROLES) == {"radiologist"}
        assert "researcher" not in INFERENCE_ROLES
        assert "clinical_auditor" not in INFERENCE_ROLES

    def test_user_create_rejects_a_too_short_password(self):
        with pytest.raises(Exception):
            UserCreate(email="alice@hospital.org", password="short1!", full_name="Alice")

    def test_user_create_rejects_a_password_without_a_special_character(self):
        with pytest.raises(Exception):
            UserCreate(email="alice@hospital.org", password="password123", full_name="Alice")

    def test_user_create_rejects_a_password_without_a_digit(self):
        with pytest.raises(Exception):
            UserCreate(email="alice@hospital.org", password="password!!!", full_name="Alice")

    def test_user_create_rejects_an_invalid_email(self):
        with pytest.raises(Exception):
            UserCreate(email="not-an-email", password="password123!", full_name="Alice")

    def test_user_login_accepts_email_in_the_username_field(self):
        login = UserLogin(username="alice@hospital.org", password="hunter2R0cks!")
        assert login.username == "alice@hospital.org"

    def test_user_response_excludes_password_fields(self):
        assert "password" not in UserResponse.model_fields
        assert "hashed_password" not in UserResponse.model_fields

    def test_token_defaults_to_bearer_type(self):
        user = UserResponse(
            id="abc123", email="alice@hospital.org", full_name="Alice Smith", role="radiologist",
            created_at="2026-01-01T00:00:00+00:00", is_active=True,
        )
        token = Token(access_token="fake.jwt.token", user=user)
        assert token.token_type == "bearer"
        assert token.expires_in_minutes == 60


# --------------------------------------------------------------------------- #
# 5. FastAPI dependencies: get_current_user / require_role
# --------------------------------------------------------------------------- #

class TestDependencies:
    def test_get_current_user_resolves_a_valid_token(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(email="frank@hospital.org", role="radiologist"))
        token = create_access_token(data={"sub": stored.email, "user_id": stored.id, "role": stored.role})

        user = auth_module.get_current_user(token=token)
        assert user.email == "frank@hospital.org"
        assert user.role == "radiologist"

    def test_get_current_user_rejects_a_token_for_a_deleted_user(self, isolated_store: UserRepository):
        token = create_access_token(data={"sub": "ghost@hospital.org", "user_id": "ghost-id", "role": "radiologist"})
        with pytest.raises(Exception) as exc_info:
            auth_module.get_current_user(token=token)
        assert exc_info.value.status_code == 401

    def test_get_current_user_rejects_a_token_for_a_deactivated_user(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(email="ivy@hospital.org", role="radiologist"))
        with isolated_store._session_factory() as session:
            db_user = session.get(User, stored.id)
            db_user.is_active = False
            session.commit()
        token = create_access_token(data={"sub": stored.email, "user_id": stored.id, "role": stored.role})
        with pytest.raises(Exception) as exc_info:
            auth_module.get_current_user(token=token)
        assert exc_info.value.status_code == 401

    def test_require_role_allows_a_permitted_role(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(email="gina@hospital.org", role="radiologist"))
        dependency = auth_module.require_role(["radiologist"])
        current_user = stored.to_response()

        result = dependency(current_user=current_user)
        assert result is current_user

    def test_require_role_rejects_a_disallowed_role_with_403(self, isolated_store: UserRepository):
        stored = isolated_store.create_user(_make_user_create(email="hank@hospital.org", role="clinical_auditor"))
        dependency = auth_module.require_role(["radiologist"])

        with pytest.raises(Exception) as exc_info:
            dependency(current_user=stored.to_response())
        assert exc_info.value.status_code == 403


# --------------------------------------------------------------------------- #
# 6. End-to-end via qknee.api.server's /api/v1/auth/* endpoints and RBAC
# --------------------------------------------------------------------------- #

class TestAuthEndpointsAndRouteProtection:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch):
        from fastapi.testclient import TestClient

        import qknee.api.server as server_module

        monkeypatch.setattr(auth_module, "user_store", _make_isolated_repository())
        return TestClient(server_module.app)

    def test_register_returns_201_and_never_leaks_the_password(self, client):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "ivy@hospital.org", "password": "password123!", "full_name": "Ivy Nguyen", "role": "radiologist"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["email"] == "ivy@hospital.org"
        assert body["role"] == "radiologist"
        assert "password" not in body
        assert "hashed_password" not in body

    def test_register_duplicate_email_returns_409(self, client):
        payload = {"email": "jack@hospital.org", "password": "password123!", "full_name": "Jack"}
        client.post("/api/v1/auth/register", json=payload)
        response = client.post("/api/v1/auth/register", json=payload)
        assert response.status_code == 409

    def test_register_without_role_defaults_to_researcher(self, client):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "kim@hospital.org", "password": "password123!", "full_name": "Kim"},
        )
        assert response.status_code == 201
        assert response.json()["role"] == "researcher"

    def test_register_rejects_a_weak_password(self, client):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "weak@hospital.org", "password": "alllowercase", "full_name": "Weak"},
        )
        assert response.status_code == 422

    def test_login_returns_bearer_token_and_user_metadata(self, client):
        client.post(
            "/api/v1/auth/register",
            json={"email": "liam@hospital.org", "password": "password123!", "full_name": "Liam", "role": "researcher"},
        )
        response = client.post("/api/v1/auth/login", json={"username": "liam@hospital.org", "password": "password123!"})

        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["user"]["email"] == "liam@hospital.org"
        assert body["user"]["role"] == "researcher"
        assert len(body["access_token"].split(".")) == 3  # header.payload.signature

    def test_login_with_wrong_password_returns_401(self, client):
        client.post(
            "/api/v1/auth/register",
            json={"email": "mia@hospital.org", "password": "password123!", "full_name": "Mia"},
        )
        response = client.post("/api/v1/auth/login", json={"username": "mia@hospital.org", "password": "wrong-pw1!"})
        assert response.status_code == 401

    def test_me_returns_the_authenticated_profile(self, client):
        client.post(
            "/api/v1/auth/register",
            json={"email": "nina@hospital.org", "password": "password123!", "full_name": "Nina", "role": "radiologist"},
        )
        token = client.post(
            "/api/v1/auth/login", json={"username": "nina@hospital.org", "password": "password123!"},
        ).json()["access_token"]

        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["email"] == "nina@hospital.org"

    def test_me_without_a_token_returns_401(self, client):
        assert client.get("/api/v1/auth/me").status_code == 401

    def test_predict_without_a_token_returns_401(self, client):
        response = client.post("/predict", files={"file": ("slice.npy", b"irrelevant")})
        assert response.status_code == 401

    def test_explain_without_a_token_returns_401(self, client):
        response = client.post("/explain", files={"file": ("slice.npy", b"irrelevant")})
        assert response.status_code == 401

    def test_predict_with_researcher_token_returns_403(self, client):
        client.post(
            "/api/v1/auth/register",
            json={"email": "oscar@hospital.org", "password": "password123!", "full_name": "Oscar", "role": "researcher"},
        )
        token = client.post(
            "/api/v1/auth/login", json={"username": "oscar@hospital.org", "password": "password123!"},
        ).json()["access_token"]

        response = client.post(
            "/predict", files={"file": ("slice.npy", b"irrelevant")}, headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403

    def test_predict_with_clinical_auditor_token_returns_403(self, client):
        client.post(
            "/api/v1/auth/register",
            json={"email": "penny@hospital.org", "password": "password123!", "full_name": "Penny", "role": "clinical_auditor"},
        )
        token = client.post(
            "/api/v1/auth/login", json={"username": "penny@hospital.org", "password": "password123!"},
        ).json()["access_token"]

        response = client.post(
            "/predict", files={"file": ("slice.npy", b"irrelevant")}, headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403

    def test_predict_with_a_radiologist_token_passes_the_auth_gate(self, client):
        """Proves the RBAC gate lets the radiologist role through to the
        actual inference logic — the request still fails downstream (422,
        bad .npy content), but critically NOT with 401/403, proving auth
        passed before file parsing ran."""
        client.post(
            "/api/v1/auth/register",
            json={"email": "user_radiologist@hospital.org", "password": "password123!", "full_name": "Rad", "role": "radiologist"},
        )
        token = client.post(
            "/api/v1/auth/login", json={"username": "user_radiologist@hospital.org", "password": "password123!"},
        ).json()["access_token"]

        response = client.post(
            "/predict", files={"file": ("slice.npy", b"not a real npy file")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code not in (401, 403)
