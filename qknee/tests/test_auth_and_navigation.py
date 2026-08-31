"""
QA regression suite: the authentication flow (`qknee.api.auth`'s
`/api/v1/auth/*` FastAPI routes) and Streamlit view-routing state
(`qknee.ui.auth_view` / `qknee.ui.dashboard`), exercised end to end.

Written against the real, deployed contracts rather than this file's own
task-description shorthand, since those are what actually protects
production:
    - Registration succeeds with `201 Created` (not `200`) and rejects a
      duplicate username with `409 Conflict` (not `400`) — `qknee.api.auth`'s
      real status codes; the docstring on each test below notes the exact
      mapping back to this suite's nominal spec.
    - The API's identity handle is `UserCreate.username` (there is no
      `email` field) — tests below use email-shaped strings as the
      human-readable test data, mapped onto valid usernames, exactly the
      way `qknee.ui.auth_view._derive_username_from_email` does for the
      real signup form.
    - Login success returns `200` with a JWT bearer token; invalid
      credentials return `401`.
    - `/predict`/`/explain` (the protected diagnostic-inference routes)
      reject any request without a valid bearer token with `401`, and a
      valid-but-wrong-role token with `403`.
    - Forged/tampered/expired JWTs are rejected with `401` and a specific,
      non-leaky `detail` string — never silently accepted, never a raw
      stack trace surfaced to the client.
    - `qknee.ui.auth_view`/`qknee.ui.dashboard`'s view-routing session
      state defaults an unauthenticated visitor to the landing/login view
      and routes an authenticated session into the diagnostic-console
      workspace.

Every FastAPI-level test gets an isolated `UserStore` (a fresh temp-file
JSON store swapped in for the shared `qknee.api.auth.user_store`
singleton), so this suite never shares or leaks state through the real
`qknee/api/users.json` — same isolation convention as
`qknee/tests/test_auth.py`.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

import jwt
from fastapi.testclient import TestClient

import qknee.api.auth as auth_module
import qknee.api.server as server_module

pytestmark = [pytest.mark.slow]


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A `TestClient` for the real `qknee.api.server` app, with
    `qknee.api.auth.user_store` swapped for a fresh, isolated JSON file so
    no test in this module reads/writes the real `qknee/api/users.json`."""
    monkeypatch.setattr(auth_module, "user_store", auth_module.UserStore(tmp_path / "users.json"))
    return TestClient(server_module.app)


def _signup(client: TestClient, username: str, password: str = "correct-password-123", role: str = "radiologist"):
    return client.post("/api/v1/auth/signup", json={"username": username, "password": password, "role": role})


def _login(client: TestClient, username: str, password: str = "correct-password-123"):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# 1. FastAPI Auth Route Testing
# --------------------------------------------------------------------------- #

class TestUserRegistration:
    def test_registration_with_valid_credentials_succeeds(self, client: TestClient):
        """Nominal spec: "status_code=200". Real contract: FastAPI's
        convention (and this router's explicit `status_code=` on
        `@router.post("/signup", ...)`) is `201 Created` for a resource
        that was just created — asserted against the real value, not the
        shorthand `200`."""
        response = _signup(client, "dr.jane.doe")  # email-shaped identifier, sanitized to a valid username

        assert response.status_code == 201
        body = response.json()
        assert body["username"] == "dr.jane.doe"
        assert body["role"] == "radiologist"
        assert "password" not in body
        assert "hashed_password" not in body

    def test_registration_with_duplicate_username_is_rejected(self, client: TestClient):
        """Nominal spec: "status_code=400". Real contract: a duplicate
        identity is a `409 Conflict` (resource already exists), which is
        the more precise HTTP semantic for this failure than a generic
        `400 Bad Request` — asserted against the real value."""
        _signup(client, "dr.jane.doe")

        response = _signup(client, "dr.jane.doe")

        assert response.status_code == 409
        assert "already taken" in response.json()["detail"].lower()

    def test_registration_duplicate_check_is_case_insensitive(self, client: TestClient):
        _signup(client, "dr.jane.doe")

        response = _signup(client, "Dr.Jane.Doe")

        assert response.status_code == 409

    def test_registration_with_an_invalid_role_is_rejected(self, client: TestClient):
        response = _signup(client, "dr.jane.doe", role="super_admin")
        assert response.status_code == 422

    def test_registration_with_too_short_a_password_is_rejected(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/signup", json={"username": "dr.jane.doe", "password": "short", "role": "radiologist"},
        )
        assert response.status_code == 422


class TestLogin:
    def test_login_with_correct_credentials_returns_200_and_a_jwt(self, client: TestClient):
        _signup(client, "dr.jane.doe", password="correct-password-123")

        response = _login(client, "dr.jane.doe", password="correct-password-123")

        assert response.status_code == 200
        body = response.json()
        assert body["token_type"] == "bearer"
        assert body["user"]["username"] == "dr.jane.doe"

        token = body["access_token"]
        assert isinstance(token, str) and token.count(".") == 2  # header.payload.signature
        decoded = jwt.decode(token, options={"verify_signature": False})
        assert decoded["sub"] == "dr.jane.doe"
        assert decoded["role"] == "radiologist"

    def test_login_with_incorrect_password_returns_401(self, client: TestClient):
        _signup(client, "dr.jane.doe", password="correct-password-123")

        response = _login(client, "dr.jane.doe", password="wrong-password")

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect username or password"

    def test_login_with_unregistered_username_returns_401(self, client: TestClient):
        response = _login(client, "nobody-registered", password="whatever-password")

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect username or password"

    def test_login_401_does_not_reveal_whether_the_username_exists(self, client: TestClient):
        """The `detail` string for "wrong password" and "unknown
        username" must be identical — a differing message would let a
        caller enumerate valid usernames via the login endpoint."""
        _signup(client, "dr.jane.doe", password="correct-password-123")

        wrong_password = _login(client, "dr.jane.doe", password="wrong-password")
        unknown_user = _login(client, "totally-unregistered-user", password="whatever")

        assert wrong_password.json()["detail"] == unknown_user.json()["detail"]


class TestProtectedInferenceRoutesRejectMissingAuth:
    """`/predict`/`/explain` are the protected diagnostic-inference routes
    (`qknee.api.auth.require_role(INFERENCE_ROLES)`)."""

    @pytest.mark.parametrize("route", ["/predict", "/explain"])
    def test_route_without_authorization_header_returns_401(self, client: TestClient, route: str):
        response = client.post(route, files={"file": ("slice.npy", b"irrelevant-content")})

        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"

    @pytest.mark.parametrize("route", ["/predict", "/explain"])
    def test_route_with_malformed_authorization_header_returns_401(self, client: TestClient, route: str):
        """A header present but not in the `Bearer <token>` scheme FastAPI's
        `OAuth2PasswordBearer` expects."""
        response = client.post(
            route, files={"file": ("slice.npy", b"irrelevant-content")},
            headers={"Authorization": "NotBearer sometoken"},
        )
        assert response.status_code == 401

    @pytest.mark.parametrize("route", ["/predict", "/explain"])
    def test_route_with_a_guest_demo_token_returns_403(self, client: TestClient, route: str):
        """A *valid, unexpired* token for a role outside `INFERENCE_ROLES`
        — proves the route distinguishes "not authenticated" (401) from
        "authenticated but not permitted" (403)."""
        _signup(client, "judge_evaluator", password="correct-password-123", role="guest_demo")
        token = _login(client, "judge_evaluator", password="correct-password-123").json()["access_token"]

        response = client.post(
            route, files={"file": ("slice.npy", b"irrelevant-content")}, headers=_bearer(token),
        )

        assert response.status_code == 403

    @pytest.mark.parametrize("role", ["radiologist", "triage_nurse"])
    def test_predict_with_a_clinical_role_token_passes_the_auth_gate(self, client: TestClient, role: str):
        """A valid, role-eligible token reaches the actual inference
        logic — the request may still fail downstream (422, unparseable
        `.npy` content), but critically NOT with 401/403, proving auth
        succeeded before file parsing ran."""
        _signup(client, f"user_{role}", password="correct-password-123", role=role)
        token = _login(client, f"user_{role}", password="correct-password-123").json()["access_token"]

        response = client.post(
            "/predict", files={"file": ("slice.npy", b"not a real npy file")}, headers=_bearer(token),
        )

        assert response.status_code not in (401, 403)

    def test_me_endpoint_without_authorization_header_returns_401(self, client: TestClient):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401


# --------------------------------------------------------------------------- #
# 2. Token Lifecycle & Expiration
# --------------------------------------------------------------------------- #

class TestForgedAndExpiredTokens:
    def test_forged_token_signed_with_a_different_secret_is_rejected(self, client: TestClient):
        forged = jwt.encode(
            {"sub": "dr.jane.doe", "role": "radiologist"},
            "a-completely-different-secret-key-not-the-servers",
            algorithm="HS256",
        )

        response = client.get("/api/v1/auth/me", headers=_bearer(forged))

        assert response.status_code == 401
        assert response.json()["detail"] == "Could not validate credentials"

    def test_tampered_token_payload_is_rejected(self, client: TestClient):
        _signup(client, "dr.jane.doe", password="correct-password-123")
        token = _login(client, "dr.jane.doe", password="correct-password-123").json()["access_token"]
        header, payload, signature = token.split(".")

        # Flip a character in the middle of the payload segment — not the
        # token's trailing character, which can land on an unused base64url
        # padding bit and decode back to the identical byte, making the
        # tamper a no-op.
        mid = len(payload) // 2
        flipped = "A" if payload[mid] != "A" else "B"
        tampered_payload = payload[:mid] + flipped + payload[mid + 1 :]
        tampered_token = ".".join([header, tampered_payload, signature])

        response = client.get("/api/v1/auth/me", headers=_bearer(tampered_token))

        assert response.status_code == 401
        assert response.json()["detail"] == "Could not validate credentials"

    def test_expired_token_is_rejected_with_expired_detail(self, client: TestClient):
        _signup(client, "dr.jane.doe", password="correct-password-123")
        expired_token = auth_module.create_access_token(
            username="dr.jane.doe", role="radiologist", expires_delta=timedelta(seconds=-1),
        )

        response = client.get("/api/v1/auth/me", headers=_bearer(expired_token))

        assert response.status_code == 401
        assert response.json()["detail"] == "Access token has expired"

    def test_token_missing_required_claims_is_rejected(self, client: TestClient):
        malformed = jwt.encode({"sub": "dr.jane.doe"}, auth_module._SECRET_KEY, algorithm="HS256")  # no "role" claim

        response = client.get("/api/v1/auth/me", headers=_bearer(malformed))

        assert response.status_code == 401
        assert response.json()["detail"] == "Access token is missing required claims"

    def test_valid_token_for_a_since_deleted_user_is_rejected(self, client: TestClient):
        """A structurally valid, unexpired, correctly-signed token whose
        subject no longer exists in the store (e.g. the account was
        removed after the token was issued)."""
        token = auth_module.create_access_token(username="ghost_user_never_created", role="radiologist")

        response = client.get("/api/v1/auth/me", headers=_bearer(token))

        assert response.status_code == 401
        assert response.json()["detail"] == "User for this access token no longer exists"

    def test_expired_token_is_also_rejected_on_the_predict_route(self, client: TestClient):
        """The expiry check applies uniformly across every protected
        route, not just `/me` — spot-checked here on the actual
        diagnostic-inference endpoint."""
        _signup(client, "dr.jane.doe", password="correct-password-123", role="radiologist")
        expired_token = auth_module.create_access_token(
            username="dr.jane.doe", role="radiologist", expires_delta=timedelta(seconds=-1),
        )

        response = client.post(
            "/predict", files={"file": ("slice.npy", b"irrelevant")}, headers=_bearer(expired_token),
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Access token has expired"

    def test_forged_token_never_returns_an_unhandled_server_error(self, client: TestClient):
        """A forged/garbage bearer token must degrade to a clean 401, never
        a 500 from an unhandled `jwt`/parsing exception leaking internals."""
        response = client.get("/api/v1/auth/me", headers=_bearer("not.a.jwt-at-all"))

        assert response.status_code == 401
        assert response.status_code != 500


# --------------------------------------------------------------------------- #
# 3. Streamlit Navigation State Smoke Test
# --------------------------------------------------------------------------- #

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
_DASHBOARD_PATH = str(Path(__file__).resolve().parents[2] / "qknee" / "ui" / "dashboard.py")


def _state(at, key: str, default=None):
    """`AppTest`'s `session_state` proxy raises for an unset key rather
    than supporting `.get()` — this wraps that. Notably,
    `qknee.ui.auth_view._log_out` `pop()`s `authenticated`/`token`/
    `user_info` entirely rather than resetting them to a falsy value, so a
    logged-out session's `"authenticated"` key may be genuinely absent."""
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


class TestNavigationStateRouting:
    def test_a_fresh_unauthenticated_session_defaults_to_the_landing_page(self):
        """On a brand-new session, `dashboard.main()` reads
        `current_page` via `st.session_state.get(..., PAGE_LANDING)` — the
        default is never written back until some navigation actually
        happens, so the key itself may still be absent; what matters is
        the *rendered* page, asserted via the landing page's own buttons."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)

        at.run()

        assert not at.exception
        assert _state(at, "current_page", "landing") == "landing"
        assert _state(at, "authenticated", False) is False
        button_labels = {b.label for b in at.button}
        assert {"Home", "Sign In", "Get Started"} <= button_labels

    def test_clicking_sign_in_transitions_to_the_login_view(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        signin_button = next(b for b in at.button if b.label == "Sign In")

        signin_button.click().run()

        assert not at.exception
        assert at.session_state["current_page"] == "login"
        assert [t.label for t in at.tabs] == ["🔐 Sign In", "📝 Create Account"]

    def test_unauthenticated_attempt_to_reach_diagnostic_console_defaults_to_login(self):
        """An unauthenticated visitor trying to reach the diagnostic
        console (via the landing page's primary CTA) is routed to the
        login view, never straight into the workspace."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        launch_button = next(b for b in at.button if "Launch Live Diagnostic" in b.label)

        launch_button.click().run()

        assert not at.exception
        assert at.session_state["current_page"] == "login"
        assert _state(at, "authenticated", False) is False

    def test_authenticated_session_routes_to_the_diagnostic_console(self):
        """After a successful (demo-account) sign-in, the session lands in
        the `workspace` state with the Diagnostic tab active — the
        authenticated equivalent of the previous test's login redirect."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if b.label == "Sign In").click().run()
        demo_button = next(b for b in at.button if "Demo Account" in b.label)

        demo_button.click().run()

        assert not at.exception
        assert at.session_state["authenticated"] is True
        assert at.session_state["current_page"] == "workspace"
        assert at.tabs[0].label == "🔬 Diagnostic View"

    def test_logging_out_of_an_authenticated_session_returns_to_landing(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if b.label == "Sign In").click().run()
        next(b for b in at.button if "Demo Account" in b.label).click().run()
        logout_button = next(b for b in at.button if b.label == "Log Out")

        logout_button.click().run()

        assert not at.exception
        assert _state(at, "authenticated", False) is False
        assert at.session_state["current_page"] == "landing"
        assert any(b.label == "Sign In" for b in at.button)
