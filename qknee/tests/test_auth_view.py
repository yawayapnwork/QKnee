"""
Tests for `qknee.ui.auth_view` (sign-in/register forms, top nav, and the
`st.session_state` auth contract wired into `qknee.ui.dashboard.main()`).

Covers:
    1. Pure helpers (institutional-email validation, field validation,
       role mapping, login-attempt throttling, session application) in
       isolation.
    2. Rendering + interaction via Streamlit's `AppTest` harness: the nav
       bar's logged-out vs. logged-in variants, the login/register forms,
       throttling lockout, and the demo-account local fallback when no
       API is reachable.

Every `AppTest` test runs with `$QKNEE_API_URL` unset/unreachable (no live
FastAPI server in this test session) — see `qknee/tests/test_api_server.py`
and `qknee/tests/test_auth.py` for the real-backend-integrated auth
contract; the point here is verifying this module's own logic (forms,
state, throttling, local fallback), not re-testing the API.

NOTE on the AppTest section: this rewrite also fixes several assertions
that were already stale against `qknee.ui.landing_page`'s/
`qknee.ui.dashboard`'s actual current button/tab labels before this auth
rewrite even started (verified via `git stash` against the pre-rewrite
tree — e.g. no "Get Started"/"Sign In"/"Log Out" widget ever existed with
those exact labels). The real CTA into the login page from the landing
page is "Launch Workstation →" (`qknee_orthoc_launch_nav`) or "Evaluate
Scans Now" (`qknee_orthoc_hero_cta`) while unauthenticated, and the real
top-nav sign-in entry point is "Clinician Portal".
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
    configured, so regular sign-in/registration deterministically hit the
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

    def test_exposes_the_spec_literal_session_state_keys(self):
        assert auth_view.SPEC_TOKEN_KEY == "auth_token"
        assert auth_view.SPEC_USER_KEY == "current_user"

    def test_current_page_values_match_the_required_set(self):
        assert {auth_view.PAGE_LANDING, auth_view.PAGE_LOGIN, auth_view.PAGE_SIGNUP, auth_view.PAGE_WORKSPACE} == {
            "landing", "login", "signup", "workspace",
        }

    def test_ui_roles_match_the_backend_role_scheme(self):
        assert auth_view.UI_ROLES == ("Radiologist", "Clinical Researcher", "Clinical Auditor")

    def test_every_ui_role_maps_to_a_valid_backend_role(self):
        valid_backend_roles = {"radiologist", "researcher", "clinical_auditor"}
        assert set(auth_view.UI_ROLE_TO_BACKEND_ROLE.values()) == valid_backend_roles
        assert set(auth_view.UI_ROLE_TO_BACKEND_ROLE.keys()) == set(auth_view.UI_ROLES)


class TestIsInstitutionalEmail:
    def test_accepts_a_dot_edu_domain(self):
        assert auth_view._is_institutional_email("jane.doe@university.edu") is True

    def test_accepts_a_dot_org_hospital_domain(self):
        assert auth_view._is_institutional_email("jane.doe@hospital.org") is True

    def test_accepts_a_domain_containing_clinic(self):
        assert auth_view._is_institutional_email("jane.doe@cityclinic.net") is True

    def test_rejects_a_free_webmail_domain(self):
        assert auth_view._is_institutional_email("jane.doe@gmail.com") is False

    def test_rejects_a_malformed_email(self):
        assert auth_view._is_institutional_email("not-an-email") is False


class TestValidateRegisterFields:
    def test_accepts_valid_fields(self):
        errors = auth_view._validate_register_fields("Jane Doe", "jane@hospital.org", "longEnough1!", "longEnough1!")
        assert errors == []

    def test_rejects_empty_full_name(self):
        errors = auth_view._validate_register_fields("", "jane@hospital.org", "longEnough1!", "longEnough1!")
        assert any("name" in e.lower() for e in errors)

    def test_rejects_a_non_institutional_email(self):
        errors = auth_view._validate_register_fields("Jane Doe", "jane@gmail.com", "longEnough1!", "longEnough1!")
        assert any("institutional" in e.lower() for e in errors)

    def test_rejects_a_too_short_password(self):
        errors = auth_view._validate_register_fields("Jane Doe", "jane@hospital.org", "sh0rt!", "sh0rt!")
        assert any("password" in e.lower() for e in errors)

    def test_rejects_mismatched_passwords(self):
        errors = auth_view._validate_register_fields("Jane Doe", "jane@hospital.org", "longEnough1!", "different1!")
        assert any("match" in e.lower() for e in errors)


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
    def test_populates_the_session_state_contract(self):
        auth_view._apply_authenticated_session({
            "access_token": "fake.jwt.token",
            "user": {"email": "jdoe@hospital.org", "role": "radiologist", "full_name": "Jane Doe"},
        })
        import streamlit as st

        assert st.session_state[auth_view.AUTHENTICATED_KEY] is True
        assert st.session_state[auth_view.TOKEN_KEY] == "fake.jwt.token"
        assert st.session_state[auth_view.USER_INFO_KEY]["email"] == "jdoe@hospital.org"
        assert st.session_state[auth_view.USER_INFO_KEY]["role"] == "radiologist"

    def test_also_populates_the_spec_literal_keys(self):
        auth_view._apply_authenticated_session({
            "access_token": "fake.jwt.token",
            "user": {"email": "jdoe@hospital.org", "role": "radiologist", "full_name": "Jane Doe"},
        })
        import streamlit as st

        assert st.session_state[auth_view.SPEC_TOKEN_KEY] == "fake.jwt.token"
        assert st.session_state[auth_view.SPEC_USER_KEY]["user_name"] == "Jane Doe"
        assert st.session_state[auth_view.SPEC_USER_KEY]["role"] == "radiologist"

    def test_never_stores_a_password_field(self):
        auth_view._apply_authenticated_session({
            "access_token": "fake.jwt.token",
            "user": {"email": "jdoe@hospital.org", "role": "radiologist", "password": "should-never-appear"},
        })
        import streamlit as st

        assert "password" not in st.session_state[auth_view.USER_INFO_KEY]
        assert "password" not in st.session_state[auth_view.SPEC_USER_KEY]


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

    def test_false_for_researcher(self):
        import streamlit as st

        st.session_state[auth_view.AUTHENTICATED_KEY] = True
        st.session_state[auth_view.USER_INFO_KEY] = {"role": "researcher"}
        assert auth_view.can_run_inference() is False

    def test_false_for_clinical_auditor(self):
        import streamlit as st

        st.session_state[auth_view.AUTHENTICATED_KEY] = True
        st.session_state[auth_view.USER_INFO_KEY] = {"role": "clinical_auditor"}
        assert auth_view.can_run_inference() is False


class TestLocalDemoSessionPayload:
    def test_is_clearly_labeled_as_not_a_real_jwt(self):
        payload = auth_view._local_demo_session_payload()
        assert "not-a-real-jwt" in payload["access_token"]

    def test_carries_the_researcher_role(self):
        payload = auth_view._local_demo_session_payload()
        assert payload["user"]["role"] == auth_view.DEMO_ROLE == "researcher"


# --------------------------------------------------------------------------- #
# 2. Rendering + interaction via Streamlit's AppTest harness
# --------------------------------------------------------------------------- #

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
_DASHBOARD_PATH = str(Path(__file__).resolve().parents[2] / "qknee" / "ui" / "dashboard.py")


@pytest.fixture
def live_auth_api(monkeypatch: pytest.MonkeyPatch):
    """Runs a real `qknee.api.server` (uvicorn, in a background thread) on
    a free local port, pointed at an isolated in-memory-SQLite user store,
    and yields its base URL. Only used by the one test that needs genuine
    401-vs-service-unreachable distinction (throttling must not count a
    down backend as a wrong password)."""
    import socket
    import threading

    import uvicorn
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import qknee.api.auth as auth_module

    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True,
    )
    auth_module.Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    monkeypatch.setattr(auth_module, "user_store", auth_module.UserRepository(session_factory=session_factory))

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


def _click_clinician_portal(at):
    next(b for b in at.button if b.key == "qknee_nav_signin").click().run()


class TestTopNavRendering:
    def test_logged_out_nav_shows_the_amber_view_only_banner(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()

        assert not at.exception
        assert any("View-Only Mode" in md.value for md in at.markdown)
        assert any(b.key == "qknee_nav_signin" for b in at.button)

    def test_logged_in_nav_shows_the_clinician_badge(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        _click_clinician_portal(at)
        next(b for b in at.button if "Demo Account" in b.label).click().run()

        assert not at.exception
        assert any("Clinical Researcher" in md.value for md in at.markdown)
        assert any(b.key == "qknee_nav_account" for b in at.button)


class TestLoginFlow:
    def test_missing_fields_shows_an_error(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        _click_clinician_portal(at)
        submit = next(b for b in at.button if "FormSubmitter" in b.key and "Authenticate Credentials" in b.label)

        submit.click().run()

        assert not at.exception
        assert any("Enter both" in e.value for e in at.error)
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False

    def test_wrong_credentials_show_an_error_and_do_not_authenticate(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        _click_clinician_portal(at)
        at.text_input(key="qknee_login_email").set_value("nobody@hospital.org")
        at.text_input(key="qknee_login_password").set_value("wrong-password!1")
        submit = next(b for b in at.button if "FormSubmitter" in b.key and "Authenticate Credentials" in b.label)

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
            f"{live_auth_api}/api/v1/auth/register",
            json={"email": "throttle_test_user@hospital.org", "password": "correct-password-123!",
                  "full_name": "Throttle Test", "role": "researcher"},
            timeout=5,
        )

        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        _click_clinician_portal(at)

        for _ in range(auth_view.MAX_LOGIN_ATTEMPTS):
            at.text_input(key="qknee_login_email").set_value("throttle_test_user@hospital.org")
            at.text_input(key="qknee_login_password").set_value("wrong-password!1")
            submit = next(b for b in at.button if "FormSubmitter" in b.key and "Authenticate Credentials" in b.label)
            submit.click().run()

        # The lockout banner (rendered at the top of `render_login_tab`,
        # before the form) reflects state as of the START of a script run
        # — the 5th failed submission's own run only *sets* the lockout
        # mid-script, so it first becomes visible on the following rerun.
        at.run()

        assert not at.exception
        assert any("Too many failed" in e.value for e in at.error)


class TestDemoAccountFlow:
    def test_demo_login_authenticates_a_researcher_session(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        _click_clinician_portal(at)
        demo_button = next(b for b in at.button if "Demo Account" in b.label)

        demo_button.click().run()

        assert not at.exception
        assert at.session_state["authenticated"] is True
        assert at.session_state["user_info"]["role"] == "researcher"
        assert at.session_state["current_page"] == "workspace"

    def test_demo_login_lands_on_diagnostic_tab_by_default(self):
        """No pending destination (arrived via the plain top-nav
        'Clinician Portal', not a landing-page CTA) — should default to
        the Diagnostic tab, not Benchmarks."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        _click_clinician_portal(at)
        next(b for b in at.button if "Demo Account" in b.label).click().run()

        assert not at.exception
        assert at.tabs[0].label == "Diagnostic Workstation"


class TestRegisterFlow:
    def test_invalid_fields_show_errors_without_calling_the_api(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if b.key == "qknee_orthoc_launch_nav").click().run()
        register_tab = next(t for t in at.tabs if "Institutional" in t.label)
        register_tab.text_input(key="qknee_register_email").set_value("not-an-email")
        register_tab.text_input(key="qknee_register_password").set_value("short")
        submit = next(b for b in at.button if "FormSubmitter" in b.key and "Institutional Access" in b.label)

        submit.click().run()

        assert not at.exception
        assert len(at.error) >= 2  # invalid email + short password (+ empty full name)
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False

    def test_valid_fields_report_service_unreachable_without_an_api(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if b.key == "qknee_orthoc_launch_nav").click().run()
        register_tab = next(t for t in at.tabs if "Institutional" in t.label)
        register_tab.text_input(key="qknee_register_full_name").set_value("Jane Doe")
        register_tab.text_input(key="qknee_register_email").set_value("jane@hospital.org")
        register_tab.text_input(key="qknee_register_password").set_value("longEnough1!")
        register_tab.text_input(key="qknee_register_confirm_password").set_value("longEnough1!")
        submit = next(b for b in at.button if "FormSubmitter" in b.key and "Institutional Access" in b.label)

        submit.click().run()

        assert not at.exception
        assert any("unreachable" in e.value.lower() for e in at.error)
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False


class TestLogoutFlow:
    def test_sign_out_clears_the_session_and_returns_to_landing(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        _click_clinician_portal(at)
        next(b for b in at.button if "Demo Account" in b.label).click().run()
        account_button = next(b for b in at.button if b.key == "qknee_nav_account")

        account_button.click().run()

        assert not at.exception
        assert _state(at, auth_view.AUTHENTICATED_KEY, False) is False
        assert auth_view.TOKEN_KEY not in at.session_state
        assert at.session_state["current_page"] == "landing"
        assert any(b.key == "qknee_nav_signin" for b in at.button)
