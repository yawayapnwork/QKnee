"""
Tests for `qknee.api.auth` — password hashing, JWT signing/verification,
the JSON user store, and the `get_current_user`/`require_role` FastAPI
dependencies — plus the wired-up `/api/v1/auth/*` endpoints and RBAC
guards on `/predict`/`/explain` in `qknee.api.server`.

Every test gets an isolated `UserStore` (a fresh temp-directory JSON file,
monkeypatched over the shared `qknee.api.auth.user_store` singleton) so
tests never share or leak state through the real `qknee/api/users.json`.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import jwt

import qknee.api.auth as auth_module
from qknee.api.auth import (
    DEFAULT_ROLE,
    INFERENCE_ROLES,
    ROLES,
    StoredUser,
    Token,
    TokenData,
    UserAlreadyExistsError,
    UserCreate,
    UserLogin,
    UserResponse,
    UserStore,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

pytestmark = [pytest.mark.slow]


@pytest.fixture
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UserStore:
    """A fresh `UserStore` backed by a per-test temp file, also swapped in
    for the module-level singleton so dependencies that reference
    `auth_module.user_store` directly (`get_current_user`) see it too."""
    store = UserStore(tmp_path / "users.json")
    monkeypatch.setattr(auth_module, "user_store", store)
    return store


# --------------------------------------------------------------------------- #
# 1. Password hashing (Argon2id)
# --------------------------------------------------------------------------- #

class TestPasswordHashing:
    def test_hash_password_does_not_return_the_plaintext(self):
        hashed = hash_password("correct horse battery staple")
        assert hashed != "correct horse battery staple"
        assert hashed.startswith("$argon2id$")

    def test_verify_password_accepts_the_correct_password(self):
        hashed = hash_password("s3cr3t-passw0rd")
        assert verify_password("s3cr3t-passw0rd", hashed) is True

    def test_verify_password_rejects_the_wrong_password(self):
        hashed = hash_password("s3cr3t-passw0rd")
        assert verify_password("wrong-password", hashed) is False

    def test_verify_password_rejects_a_malformed_hash_without_raising(self):
        assert verify_password("anything", "not-a-real-argon2-hash") is False

    def test_hash_password_is_salted_and_nondeterministic(self):
        hash_a = hash_password("same-password")
        hash_b = hash_password("same-password")
        assert hash_a != hash_b  # different random salts
        assert verify_password("same-password", hash_a)
        assert verify_password("same-password", hash_b)


# --------------------------------------------------------------------------- #
# 2. JWT access tokens
# --------------------------------------------------------------------------- #

class TestJWTAccessTokens:
    def test_create_and_decode_round_trips_username_and_role(self):
        token = create_access_token(username="dr_house", role="radiologist")
        token_data = decode_access_token(token)

        assert isinstance(token_data, TokenData)
        assert token_data.username == "dr_house"
        assert token_data.role == "radiologist"

    def test_token_is_signed_hs256(self):
        token = create_access_token(username="dr_house", role="radiologist")
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "HS256"

    def test_default_expiry_is_sixty_minutes(self):
        assert auth_module.ACCESS_TOKEN_EXPIRE_MINUTES == 60
        token = create_access_token(username="dr_house", role="radiologist")
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["exp"] - payload["iat"] == 60 * 60

    def test_expired_token_is_rejected(self):
        token = create_access_token(username="dr_house", role="radiologist", expires_delta=timedelta(seconds=-1))
        with pytest.raises(Exception) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_tampered_token_is_rejected(self):
        token = create_access_token(username="dr_house", role="radiologist")
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
            {"sub": "dr_house", "role": "radiologist"}, "some-other-secret-key-of-sufficient-length", algorithm="HS256",
        )
        with pytest.raises(Exception) as exc_info:
            decode_access_token(forged)
        assert exc_info.value.status_code == 401

    def test_token_missing_required_claims_is_rejected(self):
        malformed = jwt.encode({"sub": "dr_house"}, auth_module._SECRET_KEY, algorithm="HS256")  # no "role"
        with pytest.raises(Exception) as exc_info:
            decode_access_token(malformed)
        assert exc_info.value.status_code == 401


# --------------------------------------------------------------------------- #
# 3. UserStore
# --------------------------------------------------------------------------- #

class TestUserStore:
    def test_create_user_hashes_the_password(self, isolated_store: UserStore):
        stored = isolated_store.create_user(UserCreate(username="alice", password="hunter2rocks", role="radiologist"))
        assert isinstance(stored, StoredUser)
        assert stored.hashed_password != "hunter2rocks"
        assert verify_password("hunter2rocks", stored.hashed_password)

    def test_create_user_defaults_to_guest_demo_role(self, isolated_store: UserStore):
        stored = isolated_store.create_user(UserCreate(username="judge", password="hackathon_judge"))
        assert stored.role == DEFAULT_ROLE == "guest_demo"

    def test_create_user_rejects_duplicate_username_case_insensitively(self, isolated_store: UserStore):
        isolated_store.create_user(UserCreate(username="Alice", password="hunter2rocks", role="radiologist"))
        with pytest.raises(UserAlreadyExistsError):
            isolated_store.create_user(UserCreate(username="alice", password="different-password", role="radiologist"))

    def test_create_user_rejects_a_role_outside_the_allowed_set(self, isolated_store: UserStore):
        bad_user = UserCreate.model_construct(username="mallory", password="password123", role="admin")
        with pytest.raises(ValueError, match="role must be one of"):
            isolated_store.create_user(bad_user)

    def test_get_by_username_is_case_insensitive(self, isolated_store: UserStore):
        isolated_store.create_user(UserCreate(username="Bob", password="password123", role="triage_nurse"))
        assert isolated_store.get_by_username("bob") is not None
        assert isolated_store.get_by_username("BOB") is not None

    def test_get_by_username_returns_none_for_unknown_user(self, isolated_store: UserStore):
        assert isolated_store.get_by_username("nobody") is None

    def test_authenticate_succeeds_with_correct_credentials(self, isolated_store: UserStore):
        isolated_store.create_user(UserCreate(username="carol", password="correct-password", role="triage_nurse"))
        user = isolated_store.authenticate("carol", "correct-password")
        assert user is not None
        assert user.username == "carol"

    def test_authenticate_fails_with_wrong_password(self, isolated_store: UserStore):
        isolated_store.create_user(UserCreate(username="carol", password="correct-password", role="triage_nurse"))
        assert isolated_store.authenticate("carol", "wrong-password") is None

    def test_authenticate_fails_for_unknown_user(self, isolated_store: UserStore):
        assert isolated_store.authenticate("ghost", "anything") is None

    def test_store_persists_across_instances_via_the_json_file(self, tmp_path: Path):
        store_path = tmp_path / "users.json"
        UserStore(store_path).create_user(UserCreate(username="dana", password="password123", role="radiologist"))

        reopened = UserStore(store_path)
        assert reopened.get_by_username("dana") is not None

    def test_stored_user_to_response_omits_the_password_hash(self, isolated_store: UserStore):
        stored = isolated_store.create_user(UserCreate(username="erin", password="password123", role="guest_demo"))
        response = stored.to_response()
        assert isinstance(response, UserResponse)
        assert not hasattr(response, "hashed_password")


# --------------------------------------------------------------------------- #
# 4. Pydantic v2 schemas
# --------------------------------------------------------------------------- #

class TestSchemas:
    def test_roles_tuple_matches_the_required_default_roles(self):
        assert set(ROLES) == {"radiologist", "triage_nurse", "guest_demo"}

    def test_inference_roles_excludes_guest_demo(self):
        assert set(INFERENCE_ROLES) == {"radiologist", "triage_nurse"}
        assert "guest_demo" not in INFERENCE_ROLES

    def test_user_create_rejects_a_too_short_password(self):
        with pytest.raises(Exception):
            UserCreate(username="alice", password="short")

    def test_user_create_rejects_an_invalid_username_pattern(self):
        with pytest.raises(Exception):
            UserCreate(username="not a valid username!", password="password123")

    def test_user_login_accepts_bare_username_and_password(self):
        login = UserLogin(username="alice", password="hunter2rocks")
        assert login.username == "alice"

    def test_user_response_excludes_password_fields(self):
        assert "password" not in UserResponse.model_fields
        assert "hashed_password" not in UserResponse.model_fields

    def test_token_defaults_to_bearer_type(self):
        user = UserResponse(id="abc123", username="alice", role="radiologist", created_at="2026-01-01T00:00:00+00:00")
        token = Token(access_token="fake.jwt.token", user=user)
        assert token.token_type == "bearer"
        assert token.expires_in_minutes == 60


# --------------------------------------------------------------------------- #
# 5. FastAPI dependencies: get_current_user / require_role
# --------------------------------------------------------------------------- #

class TestDependencies:
    def test_get_current_user_resolves_a_valid_token(self, isolated_store: UserStore):
        stored = isolated_store.create_user(UserCreate(username="frank", password="password123", role="radiologist"))
        token = create_access_token(username=stored.username, role=stored.role)

        user = auth_module.get_current_user(token=token)
        assert user.username == "frank"
        assert user.role == "radiologist"

    def test_get_current_user_rejects_a_token_for_a_deleted_user(self, isolated_store: UserStore):
        token = create_access_token(username="ghost_user", role="radiologist")
        with pytest.raises(Exception) as exc_info:
            auth_module.get_current_user(token=token)
        assert exc_info.value.status_code == 401

    def test_require_role_allows_a_permitted_role(self, isolated_store: UserStore):
        stored = isolated_store.create_user(UserCreate(username="gina", password="password123", role="radiologist"))
        dependency = auth_module.require_role(["radiologist", "triage_nurse"])
        current_user = stored.to_response()

        result = dependency(current_user=current_user)
        assert result is current_user

    def test_require_role_rejects_a_disallowed_role_with_403(self, isolated_store: UserStore):
        stored = isolated_store.create_user(UserCreate(username="hank", password="password123", role="guest_demo"))
        dependency = auth_module.require_role(["radiologist"])

        with pytest.raises(Exception) as exc_info:
            dependency(current_user=stored.to_response())
        assert exc_info.value.status_code == 403


# --------------------------------------------------------------------------- #
# 6. End-to-end via qknee.api.server's /api/v1/auth/* endpoints and RBAC
# --------------------------------------------------------------------------- #

class TestAuthEndpointsAndRouteProtection:
    @pytest.fixture
    def client(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from fastapi.testclient import TestClient

        import qknee.api.server as server_module

        monkeypatch.setattr(auth_module, "user_store", UserStore(tmp_path / "users.json"))
        return TestClient(server_module.app)

    def test_signup_returns_201_and_never_leaks_the_password(self, client):
        response = client.post(
            "/api/v1/auth/signup",
            json={"username": "ivy", "password": "password123", "role": "radiologist"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["username"] == "ivy"
        assert body["role"] == "radiologist"
        assert "password" not in body
        assert "hashed_password" not in body

    def test_signup_duplicate_username_returns_409(self, client):
        payload = {"username": "jack", "password": "password123", "role": "guest_demo"}
        client.post("/api/v1/auth/signup", json=payload)
        response = client.post("/api/v1/auth/signup", json=payload)
        assert response.status_code == 409

    def test_signup_without_role_defaults_to_guest_demo(self, client):
        response = client.post("/api/v1/auth/signup", json={"username": "kim", "password": "password123"})
        assert response.status_code == 201
        assert response.json()["role"] == "guest_demo"

    def test_login_returns_bearer_token_and_user_metadata(self, client):
        client.post("/api/v1/auth/signup", json={"username": "liam", "password": "password123", "role": "triage_nurse"})
        response = client.post("/api/v1/auth/login", json={"username": "liam", "password": "password123"})

        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["user"]["username"] == "liam"
        assert body["user"]["role"] == "triage_nurse"
        assert len(body["access_token"].split(".")) == 3  # header.payload.signature

    def test_login_with_wrong_password_returns_401(self, client):
        client.post("/api/v1/auth/signup", json={"username": "mia", "password": "password123", "role": "guest_demo"})
        response = client.post("/api/v1/auth/login", json={"username": "mia", "password": "wrong"})
        assert response.status_code == 401

    def test_me_returns_the_authenticated_profile(self, client):
        client.post("/api/v1/auth/signup", json={"username": "nina", "password": "password123", "role": "radiologist"})
        token = client.post("/api/v1/auth/login", json={"username": "nina", "password": "password123"}).json()["access_token"]

        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["username"] == "nina"

    def test_me_without_a_token_returns_401(self, client):
        assert client.get("/api/v1/auth/me").status_code == 401

    def test_predict_without_a_token_returns_401(self, client):
        response = client.post("/predict", files={"file": ("slice.npy", b"irrelevant")})
        assert response.status_code == 401

    def test_explain_without_a_token_returns_401(self, client):
        response = client.post("/explain", files={"file": ("slice.npy", b"irrelevant")})
        assert response.status_code == 401

    def test_predict_with_guest_demo_token_returns_403(self, client):
        client.post("/api/v1/auth/signup", json={"username": "oscar", "password": "password123", "role": "guest_demo"})
        token = client.post("/api/v1/auth/login", json={"username": "oscar", "password": "password123"}).json()["access_token"]

        response = client.post(
            "/predict", files={"file": ("slice.npy", b"irrelevant")}, headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 403

    @pytest.mark.parametrize("role", ["radiologist", "triage_nurse"])
    def test_predict_with_a_clinical_role_token_passes_the_auth_gate(self, client, role):
        """Proves the RBAC gate lets clinical roles through to the actual
        inference logic — the request still fails downstream (422, bad
        .npy content), but critically NOT with 401/403, proving auth
        passed before file parsing ran."""
        client.post("/api/v1/auth/signup", json={"username": f"user_{role}", "password": "password123", "role": role})
        token = client.post(
            "/api/v1/auth/login", json={"username": f"user_{role}", "password": "password123"},
        ).json()["access_token"]

        response = client.post(
            "/predict", files={"file": ("slice.npy", b"not a real npy file")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code not in (401, 403)
