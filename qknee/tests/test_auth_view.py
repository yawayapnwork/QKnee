"""
Tests for `qknee.ui.auth_view` (login/signup forms, top nav, and the
`st.session_state` auth contract wired into `qknee.ui.dashboard.main()`).

Covers:
    1. Pure helpers (username derivation, field validation, role mapping,
       login-attempt throttling, session application) in isolation.
    2. Rendering + interaction via Streamlit's `AppTest` harness: the nav
       bar's logged-out vs. logged-in variants, the login/signup forms,
       throttling lockout, and the demo-account local fallback when no
       API is reachable.

Every `AppTest` test runs with `$QKNEE_API_URL` unset/unreachable (no live
FastAPI server in this test session) — see `qknee/tests/test_api_server.py`
and `qknee/tests/test_auth.py` for the real-backend-integrated auth
contract; the point here is verifying this module's own logic (forms,
state, throttling, local fallback), not re-testing the API.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import qknee.ui.auth_view as auth_view

pytestmark = [pytest.mark.slow]


@pytest.fixture(autouse=True)
def _no_api_url(monkeypatch: pytest.MonkeyPatch):
    """Every test in this module runs as if no `$QKNEE_API_URL` is
    configured, so regular login/signup deterministically hit the
    "service unreachable" error path and only the Demo Account button
    exercises the local fallback — matching this module's own documented
    contract (see its module docstring)."""
    monkeypatch.delenv("QKNEE_API_URL", raising=False)


# --------------------------------------------------------------------------- #
# 1. Pure helpers
# --------------------------------------------------------------------------- #

class TestModuleImports:
    def test_exposes_the_required_session_state_keys(self):
        assert auth_view.AUTHENTICATED_KEY == "authenticated"
        assert auth_view.TOKEN_KEY == "token"
        assert auth_view.USER_INFO_KEY == "user_info"
        assert auth_view.CURRENT_PAGE_KEY == "current_page"

    def test_current_page_values_match_the_required_set(self):
        assert {auth_view.PAGE_LANDING, auth_view.PAGE_LOGIN, auth_view.PAGE_SIGNUP, auth_view.PAGE_WORKSPACE} == {
            "landing", "login", "signup", "workspace",
        }

    def test_ui_roles_match_the_required_copy(self):
        assert auth_view.UI_ROLES == ("Radiologist", "Clinical Researcher", "Student Evaluator")

    def test_every_ui_role_maps_to_a_valid_backend_role(self):
        valid_backend_roles = {"radiologist", "triage_nurse", "guest_demo"}
        assert set(auth_view.UI_ROLE_TO_BACKEND_ROLE.values()) <= valid_backend_roles
        assert set(auth_view.UI_ROLE_TO_BACKEND_ROLE.keys()) == set(auth_view.UI_ROLES)


class TestDeriveUsernameFromEmail:
    def test_uses_the_local_part_of_the_email(self):
        assert auth_view._derive_username_from_email("jane.doe@hospital.org") == "jane.doe"

    def test_strips_disallowed_characters(self):
        assert auth_view._derive_username_from_email("jane+test@hospital.org") == "janetest"

    def test_pads_a_too_short_local_part(self):
        username = auth_view._derive_username_from_email("jd@hospital.org")
        assert len(username) >= 3

    def test_truncates_a_too_long_local_part(self):
        long_local_part = "a" * 100
        username = auth_view._derive_username_from_email(f"{long_local_part}@hospital.org")
        assert len(username) <= 64


class TestValidateSignupFields:
    def test_accepts_valid_fields(self):
        assert auth_view._validate_signup_fields("Jane Doe", "jane@hospital.org", "longenoughpw") == []

    def test_rejects_empty_full_name(self):
        errors = auth_view._validate_signup_fields("", "jane@hospital.org", "longenoughpw")
        assert any("name" in e.lower() for e in errors)

    def test_rejects_an_invalid_email(self):
        errors = auth_view._validate_signup_fields("Jane Doe", "not-an-email", "longenoughpw")
        assert any("email" in e.lower() for e in errors)

    def test_rejects_a_too_short_password(self):
        errors = auth_view._validate_signup_fields("Jane Doe", "jane@hospital.org", "short")
        assert any("password" in e.lower() for e in errors)


class TestLoginThrottling:
    def test_not_locked_out_before_the_threshold(self):
        for _ in range(auth_view.MAX_LOGIN_ATTEMPTS - 1):
            auth_view._register_failed_attempt()
        assert auth_view._lockout_remaining_seconds() == 0.0

    def test_locked_out_at_the_threshold(self):
        for _ in range(auth_view.MAX_LOGIN_ATTEMPTS):
            auth_view._register_failed_attempt()
        assert auth_view._lockout_remaining_seconds() > 0.0

    def test_reset_clears_the_lockout(self):
        for _ in range(auth_view.MAX_LOGIN_ATTEMPTS):
            auth_view._register_failed_attempt()
        auth_view._reset_failed_attempts()
        assert auth_view._lockout_remaining_seconds() == 0.0


class TestApplyAuthenticatedSession:
    def test_populates_all_four_session_state_keys(self):
        auth_view._apply_authenticated_session({
            "access_token": "fake.jwt.token",
            "user": {"username": "jdoe", "role": "radiologist", "full_name": "Jane Doe"},
        })
        import streamlit as st

        assert st.session_state[auth_view.AUTHENTICATED_KEY] is True
        assert st.session_state[auth_view.TOKEN_KEY] == "fake.jwt.token"
        assert st.session_state[auth_view.USER_INFO_KEY]["username"] == "jdoe"
        assert st.session_state[auth_view.USER_INFO_KEY]["role"] == "radiologist"

    def test_never_stores_a_password_field(self):
        auth_view._apply_authenticated_session({
            "access_token": "fake.jwt.token",
            "user": {"username": "jdoe", "role": "radiologist", "password": "should-never-appear"},
        })
        import streamlit as st

        assert "password" not in st.session_state[auth_view.USER_INFO_KEY]


class TestCanRunInference:
    def test_false_when_not_authenticated(self):
        import streamlit as st

        st.session_state[auth_view.AUTHENTICATED_KEY] = False
        assert auth_view.can_run_inference() is False

    def test_true_for_radiologist(self):
        import streamlit as st

        st.session_state[auth_view.AUTHENTICATED_KEY] = True
        st.session_state[auth_view.USER_INFO_KEY] = {"role": "radiologist"}
        assert auth_view.can_run_inference() is True

    def test_true_for_triage_nurse(self):
        import streamlit as st

        st.session_state[auth_view.AUTHENTICATED_KEY] = True
        st.session_state[auth_view.USER_INFO_KEY] = {"role": "triage_nurse"}
        assert auth_view.can_run_inference() is True

    def test_false_for_guest_demo(self):
        import streamlit as st

        st.session_state[auth_view.AUTHENTICATED_KEY] = True
        st.session_state[auth_view.USER_INFO_KEY] = {"role": "guest_demo"}
        assert auth_view.can_run_inference() is False


class TestLocalDemoSessionPayload:
    def test_is_clearly_labeled_as_not_a_real_jwt(self):
        payload = auth_view._local_demo_session_payload()
        assert "not-a-real-jwt" in payload["access_token"]

    def test_carries_the_guest_demo_role(self):
        payload = auth_view._local_demo_session_payload()
        assert payload["user"]["role"] == auth_view.DEMO_ROLE == "guest_demo"


# --------------------------------------------------------------------------- #
# 2. Rendering + interaction via Streamlit's AppTest harness
# --------------------------------------------------------------------------- #

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
_DASHBOARD_PATH = str(Path(__file__).resolve().parents[2] / "qknee" / "ui" / "dashboard.py")


@pytest.fixture
def live_auth_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Runs a real `qknee.api.server` (uvicorn, in a background thread) on
    a free local port, pointed at an isolated per-test user store, and
    yields its base URL. Only used by the one test that needs genuine
    401-vs-service-unreachable distinction (throttling must not count a
    down backend as a wrong password)."""
    import socket
    import threading

    import uvicorn

    import qknee.api.auth as auth_module

    monkeypatch.setattr(auth_module, "user_store", auth_module.UserStore(tmp_path / "users.json"))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    import qknee.api.server as server_module

    config = uvicorn.Config(server_module.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    import requests

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if requests.get(f"{base_url}/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.5)
    else:
        raise RuntimeError("live_auth_api server did not become healthy in time")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


def _state(at, key: str, default=None):
    """`AppTest`'s `session_state` proxy raises `KeyError`/`AttributeError`
    for an unset key rather than supporting `.get()` — this wraps that."""
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


class TestTopNavRendering:
    def test_logged_out_nav_shows_the_required_links(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()

        assert not at.exception
        labels = {b.label for b in at.button}
        assert {"Home", "Product Features", "Benchmarks", "Sign In", "Get Started"} <= labels

    def test_logged_in_nav_shows_avatar_role_and_logout(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Get Started" in b.label).click().run()
        next(b for b in at.button if "Demo Account" in b.label).click().run()

        assert not at.exception
        labels = {b.label for b in at.button}
        assert "Log Out" in labels
        assert any("Switch to Diagnostic Workspace" in label for label in labels)
        # The nav badge itself is raw HTML (st.markdown), not a widget —
        # assert on the markdown blocks instead of a button/label.
        assert any("[Student Evaluator]" in md.value for md in at.markdown)


class TestLoginFlow:
    def test_missing_fields_shows_an_error(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Sign In" == b.label).click().run()
        submit = next(b for b in at.button if "FormSubmitter" in b.key and "Sign In" in b.label)

        submit.click().run()

        assert not at.exception
        assert any("Enter both" in e.value for e in at.error)
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False

    def test_wrong_credentials_show_an_error_and_do_not_authenticate(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Sign In" == b.label).click().run()
        at.text_input(key="qknee_login_username").set_value("nobody")
        at.text_input(key="qknee_login_password").set_value("wrong-password")
        submit = next(b for b in at.button if "FormSubmitter" in b.key and "Sign In" in b.label)

        submit.click().run()

        assert not at.exception
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False
        # With no reachable API, this is a service-unreachable error, not a
        # credentials error — either way, no session gets authenticated.
        assert len(at.error) >= 1

    def test_repeated_failures_eventually_lock_out_the_form(self, live_auth_api: str, monkeypatch: pytest.MonkeyPatch):
        """Throttling only counts genuine wrong-password rejections (401
        from a reachable auth service), not "service unreachable" errors —
        so this needs a real, reachable `qknee.api.auth` backend rather
        than the module-wide `_no_api_url` fixture; a registered account
        with the wrong password is submitted `MAX_LOGIN_ATTEMPTS` times."""
        monkeypatch.setenv("QKNEE_API_URL", live_auth_api)
        import requests

        requests.post(
            f"{live_auth_api}/api/v1/auth/signup",
            json={"username": "throttle_test_user", "password": "correct-password-123", "role": "guest_demo"},
            timeout=5,
        )

        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Sign In" == b.label).click().run()

        for _ in range(auth_view.MAX_LOGIN_ATTEMPTS):
            at.text_input(key="qknee_login_username").set_value("throttle_test_user")
            at.text_input(key="qknee_login_password").set_value("wrong-password")
            submit = next(b for b in at.button if "FormSubmitter" in b.key and "Sign In" in b.label)
            submit.click().run()

        # The lockout banner (rendered at the top of `render_login_tab`,
        # before the form) reflects state as of the START of a script run
        # — the 5th failed submission's own run only *sets* the lockout
        # mid-script, so it first becomes visible on the following rerun.
        at.run()

        assert not at.exception
        assert any("Too many failed" in e.value for e in at.error)


class TestDemoAccountFlow:
    def test_demo_login_authenticates_a_guest_demo_session(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Sign In" == b.label).click().run()
        demo_button = next(b for b in at.button if "Demo Account" in b.label)

        demo_button.click().run()

        assert not at.exception
        assert at.session_state["authenticated"] is True
        assert at.session_state["user_info"]["role"] == "guest_demo"
        assert at.session_state["current_page"] == "workspace"

    def test_demo_login_lands_on_diagnostic_tab_by_default(self):
        """No pending destination (arrived via the plain top-nav 'Sign
        In', not a landing-page CTA) — should default to the Diagnostic
        tab, not Benchmarks."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Sign In" == b.label).click().run()
        next(b for b in at.button if "Demo Account" in b.label).click().run()

        assert not at.exception
        assert at.tabs[0].label == "Diagnostic Workstation"


class TestSignupFlow:
    def test_invalid_fields_show_errors_without_calling_the_api(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Get Started" in b.label).click().run()
        signup_tab = at.tabs[0]  # "Create Account" is first when arriving via Get Started
        signup_tab.text_input(key="qknee_signup_email").set_value("not-an-email")
        signup_tab.text_input(key="qknee_signup_password").set_value("short")
        submit = signup_tab.get("button")[0]

        submit.click().run()

        assert not at.exception
        assert len(at.error) >= 2  # invalid email + short password (+ empty full name)
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False

    def test_valid_fields_report_service_unreachable_without_an_api(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Get Started" in b.label).click().run()
        signup_tab = at.tabs[0]
        signup_tab.text_input(key="qknee_signup_full_name").set_value("Jane Doe")
        signup_tab.text_input(key="qknee_signup_email").set_value("jane@hospital.org")
        signup_tab.text_input(key="qknee_signup_password").set_value("longenoughpassword")
        submit = signup_tab.get("button")[0]

        submit.click().run()

        assert not at.exception
        assert any("unreachable" in e.value.lower() for e in at.error)
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False


class TestLogoutFlow:
    def test_log_out_clears_the_session_and_returns_to_landing(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Get Started" in b.label).click().run()
        next(b for b in at.button if "Demo Account" in b.label).click().run()
        logout_button = next(b for b in at.button if b.label == "Log Out")

        logout_button.click().run()

        assert not at.exception
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False
        assert auth_view.TOKEN_KEY not in at.session_state
        assert at.session_state["current_page"] == "landing"
        assert any("Get Started" in b.label for b in at.button)
