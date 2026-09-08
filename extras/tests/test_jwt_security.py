"""
Tests for AUDIT.md P1 #8: `qknee.api.auth`'s JWT secret resolution/
validation, token expiry, algorithm pinning, and auth-failure message
hygiene.

`resolve_jwt_secret` is a pure function of an env mapping (see its
docstring) — every "missing/insecure/valid secret" scenario below is
exercised by passing a plain `dict` directly, without touching real
process env vars, monkeypatching `os.environ`, or re-importing the
`qknee.api.auth` module (which would also rebuild its DB engine and is
therefore expensive/side-effectful).
"""

from __future__ import annotations

from datetime import timedelta

import jwt
import pytest

pytest.importorskip("fastapi")

import qknee.api.auth as auth_module
from qknee.api.auth import (
    InsecureJWTConfigurationError,
    create_access_token,
    decode_access_token,
    resolve_jwt_secret,
)

pytestmark = [pytest.mark.slow]

STRONG_SECRET = "a-genuinely-strong-random-secret-" + "x" * 32  # > 32 chars, no weak markers


# --------------------------------------------------------------------------- #
# 1. resolve_jwt_secret: missing / insecure / valid secret, in both
#    production and development environments.
# --------------------------------------------------------------------------- #

class TestResolveJwtSecretMissingInProduction:
    def test_no_env_at_all_raises_in_the_default_strict_mode(self):
        """QKNEE_ENV unset must behave like production -- the original
        vulnerability was exactly a permissive default nobody opted into."""
        with pytest.raises(InsecureJWTConfigurationError, match="No JWT signing secret is configured"):
            resolve_jwt_secret(env={})

    def test_explicit_production_env_with_no_secret_raises(self):
        with pytest.raises(InsecureJWTConfigurationError, match="No JWT signing secret is configured"):
            resolve_jwt_secret(env={"QKNEE_ENV": "production"})

    def test_unrecognized_env_value_is_treated_as_production_not_silently_permissive(self):
        with pytest.raises(InsecureJWTConfigurationError):
            resolve_jwt_secret(env={"QKNEE_ENV": "staging"})


class TestResolveJwtSecretInsecureDefault:
    def test_the_old_committed_default_string_is_rejected_even_if_someone_sets_it_explicitly(self):
        old_default = "INSECURE-DEV-ONLY-CHANGE-ME-VIA-QKNEE_JWT_SECRET_KEY-ENV-VAR"
        with pytest.raises(InsecureJWTConfigurationError):
            resolve_jwt_secret(env={"QKNEE_JWT_SECRET_KEY": old_default})

    def test_a_short_secret_is_rejected_in_production(self):
        with pytest.raises(InsecureJWTConfigurationError, match="too short"):
            resolve_jwt_secret(env={"QKNEE_JWT_SECRET_KEY": "short"})

    def test_a_weak_placeholder_word_is_rejected_even_if_long_enough_would_otherwise_pass(self):
        with pytest.raises(InsecureJWTConfigurationError):
            resolve_jwt_secret(env={"QKNEE_JWT_SECRET_KEY": "changeme"})

    def test_low_entropy_repeated_character_secret_is_rejected(self):
        with pytest.raises(InsecureJWTConfigurationError):
            resolve_jwt_secret(env={"QKNEE_JWT_SECRET_KEY": "x" * 40})

    def test_weak_secret_is_accepted_in_development_with_no_extra_opt_in_needed(self):
        """A configured (even if weak) secret in a declared dev environment
        is an informed choice, unlike the fully-missing case, which still
        requires the separate QKNEE_ALLOW_INSECURE_JWT_SECRET opt-in."""
        secret = resolve_jwt_secret(env={"QKNEE_ENV": "development", "QKNEE_JWT_SECRET_KEY": "short"})
        assert secret == "short"

    def test_never_logs_the_actual_secret_value(self, caplog: pytest.LogCaptureFixture):
        import logging

        weak_secret = "weak-but-configured-secret"
        with caplog.at_level(logging.WARNING, logger="qknee.api.auth"):
            resolve_jwt_secret(env={"QKNEE_ENV": "development", "QKNEE_JWT_SECRET_KEY": weak_secret})

        for record in caplog.records:
            assert weak_secret not in record.getMessage()


class TestResolveJwtSecretMissingInDevelopment:
    def test_missing_secret_in_dev_without_the_explicit_opt_in_still_raises(self):
        with pytest.raises(InsecureJWTConfigurationError, match="QKNEE_ALLOW_INSECURE_JWT_SECRET"):
            resolve_jwt_secret(env={"QKNEE_ENV": "development"})

    def test_missing_secret_in_dev_with_both_explicit_opt_ins_returns_the_fixed_dev_secret(self):
        secret = resolve_jwt_secret(env={"QKNEE_ENV": "development", "QKNEE_ALLOW_INSECURE_JWT_SECRET": "1"})
        assert secret  # non-empty
        assert "insecure" in secret.lower()

    def test_never_logs_the_dev_secret_value_either(self, caplog: pytest.LogCaptureFixture):
        import logging

        with caplog.at_level(logging.WARNING, logger="qknee.api.auth"):
            secret = resolve_jwt_secret(env={"QKNEE_ENV": "development", "QKNEE_ALLOW_INSECURE_JWT_SECRET": "1"})

        for record in caplog.records:
            assert secret not in record.getMessage()


class TestResolveJwtSecretValidConfiguredSecret:
    def test_a_strong_secret_is_accepted_in_production(self):
        assert resolve_jwt_secret(env={"QKNEE_ENV": "production", "QKNEE_JWT_SECRET_KEY": STRONG_SECRET}) == STRONG_SECRET

    def test_a_strong_secret_is_accepted_with_no_env_declared_at_all(self):
        assert resolve_jwt_secret(env={"QKNEE_JWT_SECRET_KEY": STRONG_SECRET}) == STRONG_SECRET

    def test_qknee_jwt_secret_key_takes_precedence_over_secret_key(self):
        secret = resolve_jwt_secret(
            env={"QKNEE_JWT_SECRET_KEY": STRONG_SECRET, "SECRET_KEY": "some-other-strong-secret-" + "y" * 32},
        )
        assert secret == STRONG_SECRET

    def test_generic_secret_key_env_var_is_accepted_as_a_fallback(self):
        assert resolve_jwt_secret(env={"SECRET_KEY": STRONG_SECRET}) == STRONG_SECRET

    def test_surrounding_whitespace_is_stripped(self):
        assert resolve_jwt_secret(env={"QKNEE_JWT_SECRET_KEY": f"  {STRONG_SECRET}  "}) == STRONG_SECRET


# --------------------------------------------------------------------------- #
# 1b. Password hashing configuration (AUDIT.md P1 #8 requirement 10) --
# already correct (Argon2id, OWASP-exceeding parameters); locked in here as
# a regression guard so a future refactor can't silently weaken it.
# --------------------------------------------------------------------------- #

class TestPasswordHashingConfiguration:
    def test_uses_argon2id(self):
        from argon2.low_level import Type

        assert auth_module._password_hasher.type == Type.ID

    def test_meets_or_exceeds_owasp_minimum_parameters(self):
        hasher = auth_module._password_hasher
        # OWASP Password Storage Cheat Sheet minimums for Argon2id.
        assert hasher.memory_cost >= 19 * 1024  # >= 19 MiB
        assert hasher.time_cost >= 2
        assert hasher.parallelism >= 1
        assert hasher.salt_len >= 16

    def test_hash_and_verify_round_trip(self):
        from qknee.api.auth import hash_password, verify_password

        hashed = hash_password("a-real-password-123!")
        assert verify_password("a-real-password-123!", hashed) is True
        assert verify_password("wrong-password", hashed) is False

    def test_password_hash_never_stores_plaintext(self):
        from qknee.api.auth import hash_password

        hashed = hash_password("super-secret-password-1!")
        assert "super-secret-password-1!" not in hashed


# --------------------------------------------------------------------------- #
# 2. Token expiration
# --------------------------------------------------------------------------- #

class TestTokenExpiration:
    def test_a_freshly_issued_token_decodes_successfully(self):
        token = create_access_token(data={"sub": "a@b.com", "user_id": "1", "role": "researcher"})
        decoded = decode_access_token(token)
        assert decoded.email == "a@b.com"

    def test_an_expired_token_is_rejected_with_a_401(self):
        token = create_access_token(
            data={"sub": "a@b.com", "user_id": "1", "role": "researcher"}, expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(Exception) as exc_info:
            decode_access_token(token)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Access token has expired"

    def test_default_expiry_matches_configured_access_token_expire_minutes(self):
        import jwt as pyjwt

        token = create_access_token(data={"sub": "a@b.com", "user_id": "1", "role": "researcher"})
        payload = pyjwt.decode(token, auth_module._SECRET_KEY, algorithms=["HS256"])
        lifetime_seconds = payload["exp"] - payload["iat"]
        assert lifetime_seconds == pytest.approx(auth_module.ACCESS_TOKEN_EXPIRE_MINUTES * 60, abs=2)

    def test_access_token_expire_minutes_is_within_a_sane_bound(self):
        """AUDIT.md P1 #8 requirement 8: a bearer token with no refresh/
        revocation mechanism should not be valid for an unreasonably long
        time. Guards against a future config regression, not a specific
        current bug."""
        assert 0 < auth_module.ACCESS_TOKEN_EXPIRE_MINUTES <= 24 * 60


# --------------------------------------------------------------------------- #
# 3. Invalid tokens / algorithm confusion
# --------------------------------------------------------------------------- #

class TestInvalidTokensAndAlgorithmConfusion:
    def test_garbage_token_is_rejected(self):
        with pytest.raises(Exception) as exc_info:
            decode_access_token("not-a-jwt-at-all")
        assert exc_info.value.status_code == 401

    def test_token_signed_with_a_different_secret_is_rejected(self):
        forged = jwt.encode(
            {"sub": "a@b.com", "user_id": "1", "role": "radiologist"}, "attacker-controlled-secret", algorithm="HS256",
        )
        with pytest.raises(Exception) as exc_info:
            decode_access_token(forged)
        assert exc_info.value.status_code == 401

    def test_alg_none_token_is_rejected(self):
        """The classic JWT bypass: an attacker crafts a token with
        `alg: none` and no signature at all, hoping a lenient verifier
        skips signature checking entirely."""
        forged = jwt.encode(
            {"sub": "a@b.com", "user_id": "1", "role": "radiologist"}, key=None, algorithm="none",
        )
        with pytest.raises(Exception) as exc_info:
            decode_access_token(forged)
        assert exc_info.value.status_code == 401

    def test_decode_access_token_pins_algorithms_explicitly(self):
        """Structural guarantee, not just a behavioral one: `algorithms=`
        must be a real, non-empty allowlist -- never omitted (which lets
        PyJWT trust whatever `alg` the token itself claims)."""
        assert auth_module._ALLOWED_DECODE_ALGORITHMS == ("HS256",)

    def test_token_missing_required_claims_is_rejected(self):
        malformed = jwt.encode({"sub": "a@b.com"}, auth_module._SECRET_KEY, algorithm="HS256")
        with pytest.raises(Exception) as exc_info:
            decode_access_token(malformed)
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Access token is missing required claims"


# --------------------------------------------------------------------------- #
# 4. Protected endpoint (full app, real TestClient)
# --------------------------------------------------------------------------- #

def _make_isolated_repository():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from qknee.api.auth import Base

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    from qknee.api.auth import UserRepository

    return UserRepository(session_factory)


class TestProtectedEndpoint:
    @pytest.fixture
    def client(self, monkeypatch: pytest.MonkeyPatch):
        from fastapi.testclient import TestClient

        import qknee.api.server as server_module

        monkeypatch.setattr(auth_module, "user_store", _make_isolated_repository())
        monkeypatch.setattr(auth_module.limiter, "enabled", False)
        return TestClient(server_module.app)

    def test_me_without_a_token_is_401(self, client):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401

    def test_me_with_a_valid_token_succeeds(self, client, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QKNEE_RADIOLOGIST_INVITE_CODE", "test-code")
        register = client.post(
            "/api/v1/auth/register",
            json={
                "email": "protected@hospital.org", "password": "correct-password-123!",
                "full_name": "Dr. Protected", "role": "radiologist", "invite_code": "test-code",
            },
        )
        assert register.status_code == 201
        login = client.post(
            "/api/v1/auth/login", json={"username": "protected@hospital.org", "password": "correct-password-123!"},
        )
        assert login.status_code == 200
        token = login.json()["access_token"]

        response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["email"] == "protected@hospital.org"

    def test_me_with_a_malformed_bearer_header_is_401(self, client):
        response = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
        assert response.status_code == 401

    def test_login_failure_never_reveals_whether_the_email_is_registered(self, client):
        """AUDIT.md P1 #8 requirement 11: the same generic message for a
        wrong password vs. a never-registered email."""
        wrong_password = client.post(
            "/api/v1/auth/login", json={"username": "nonexistent@hospital.org", "password": "whatever-123!"},
        )
        assert wrong_password.status_code == 401
        assert wrong_password.json()["detail"] == "Incorrect email or password"
