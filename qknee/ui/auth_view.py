"""
Q-Knee authentication UI (Streamlit) — login/signup forms, the persistent
top navigation toolbar, and the `st.session_state` contract that wires
both into `qknee.ui.dashboard.main()`.

Session-state contract (the four keys `qknee.ui.dashboard` and this module
both read/write):
    authenticated (bool) - whether the current browser session has a valid
                            (or fallback-local) authenticated session.
    token (str)          - the bearer JWT from `qknee.api.auth` (or a
                            clearly-labeled placeholder in offline/local-
                            fallback mode — see `_local_demo_session_payload`).
    user_info (dict)     - {"username", "role", "full_name", "affiliation",
                            "email", "created_at"} — never includes a
                            password or password hash.
    current_page (str)   - one of `PAGE_LANDING` / `PAGE_LOGIN` /
                            `PAGE_SIGNUP` / `PAGE_WORKSPACE`; the top-level
                            page router `dashboard.main()` reads.

Backend wiring: form submissions call the real `qknee.api.auth` endpoints
(`POST {QKNEE_API_URL}/api/v1/auth/{signup,login}`) when `$QKNEE_API_URL`
is set and reachable — same env var / reachability probe convention
`qknee.ui.dashboard.resolve_api_url`/`api_is_reachable` use for inference.
When the API is unreachable, regular username/password login and signup
surface a clear error (never silently "authenticate" an arbitrary password
against nothing — see `_login_via_api_or_error`'s docstring) — only the
"Sign in with Demo Account" button falls back to a local, clearly-labeled
guest session, so the product stays demoable end to end even with no
backend running, matching the mock-fallback convention used everywhere
else in this codebase (`qknee.ui.dashboard`, `qknee.api.server`).

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import streamlit as st

from qknee.config.logging_config import get_logger
from qknee.ui import theme

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Session-state keys (the contract `qknee.ui.dashboard.main()` reads)
# --------------------------------------------------------------------------- #

AUTHENTICATED_KEY = "authenticated"
TOKEN_KEY = "token"
USER_INFO_KEY = "user_info"
CURRENT_PAGE_KEY = "current_page"

PAGE_LANDING = "landing"
PAGE_LOGIN = "login"
PAGE_SIGNUP = "signup"
PAGE_WORKSPACE = "workspace"

# Internal-only session-state keys (login throttling).
_FAILED_ATTEMPTS_KEY = "_qknee_auth_failed_attempts"
_LOCKOUT_UNTIL_KEY = "_qknee_auth_lockout_until"

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 30.0

LOGIN_ENDPOINT = "/api/v1/auth/login"
SIGNUP_ENDPOINT = "/api/v1/auth/signup"

# --------------------------------------------------------------------------- #
# Role mapping: this UI's role selector (task-specified copy) vs.
# `qknee.api.auth.ROLES`'s three backend role values. "Clinical Researcher"
# and "Student Evaluator" have no literal backend equivalent, so they're
# mapped onto the closest existing permission tier: a Clinical Researcher
# gets the same inference-capable tier as `triage_nurse` (can run
# /predict, /explain — a researcher plausibly needs to), a Student
# Evaluator gets `guest_demo` (read-only/demo-tier — matches "evaluating
# the system" rather than "running real clinical inference").
# --------------------------------------------------------------------------- #

UI_ROLES: tuple[str, ...] = ("Radiologist", "Clinical Researcher", "Student Evaluator")
UI_ROLE_TO_BACKEND_ROLE: Dict[str, str] = {
    "Radiologist": "radiologist",
    "Clinical Researcher": "triage_nurse",
    "Student Evaluator": "guest_demo",
}
# Mirrors `qknee.api.auth.INFERENCE_ROLES` (the two roles the API's
# `require_role` guard permits onto `/predict`/`/explain`) — duplicated as
# a plain literal tuple rather than imported, since `qknee.api.auth` has
# import-time side effects (creates/touches `qknee/api/users.json`, logs
# the insecure-default-JWT-secret warning) that don't belong in the UI
# process just to read one constant.
_CLINICAL_INFERENCE_ROLES: tuple[str, ...] = ("radiologist", "triage_nurse")

DEMO_USERNAME = "guest_demo_judge"
DEMO_PASSWORD = "QKneeDemo!2026"  # noqa: S105 - intentionally public: a shared, guest_demo-role-only demo account
DEMO_ROLE = "guest_demo"
DEMO_FULL_NAME = "Demo Judge"


# --------------------------------------------------------------------------- #
# Auth-service errors
# --------------------------------------------------------------------------- #

class _AuthServiceError(Exception):
    """Base class for a failed auth call — message is shown to the user as-is."""


class _InvalidCredentialsError(_AuthServiceError):
    def __init__(self) -> None:
        super().__init__("Incorrect username or password.")


class _UsernameTakenError(_AuthServiceError):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)


# --------------------------------------------------------------------------- #
# API client (mirrors qknee.ui.dashboard.resolve_api_url/api_is_reachable's
# conventions; duplicated rather than imported to avoid a
# dashboard<->auth_view import cycle, since dashboard.py imports this module)
# --------------------------------------------------------------------------- #

def _resolve_api_url() -> Optional[str]:
    return os.environ.get("QKNEE_API_URL") or None


def _api_is_reachable(api_url: str, timeout: float = 1.5) -> bool:
    try:
        import requests

        response = requests.get(f"{api_url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception as exc:  # noqa: BLE001 - any failure just means "not reachable"
        logger.debug("Auth API health check failed for %s: %s", api_url, exc)
        return False


def _safe_error_detail(response) -> str:
    try:
        detail = response.json().get("detail")
        if detail:
            return str(detail)
    except Exception:  # noqa: BLE001 - fall through to the generic message below
        pass
    return f"HTTP {response.status_code}"


def _login_via_api(api_url: str, username: str, password: str) -> dict:
    import requests

    try:
        response = requests.post(
            f"{api_url}{LOGIN_ENDPOINT}", json={"username": username, "password": password}, timeout=5,
        )
    except requests.RequestException as exc:
        raise _AuthServiceError(f"Could not reach the authentication service ({exc}).") from exc

    if response.status_code == 401:
        raise _InvalidCredentialsError()
    if response.status_code >= 400:
        raise _AuthServiceError(_safe_error_detail(response))
    return response.json()  # {"access_token", "token_type", "expires_in_minutes", "user": {...}}


def _signup_via_api(api_url: str, payload: dict) -> dict:
    import requests

    try:
        response = requests.post(f"{api_url}{SIGNUP_ENDPOINT}", json=payload, timeout=5)
    except requests.RequestException as exc:
        raise _AuthServiceError(f"Could not reach the registration service ({exc}).") from exc

    if response.status_code == 409:
        raise _UsernameTakenError(_safe_error_detail(response))
    if response.status_code >= 400:
        raise _AuthServiceError(_safe_error_detail(response))
    return response.json()  # UserResponse: {"id", "username", "role", "created_at"}


def _login_via_api_or_error(username: str, password: str) -> dict:
    """Regular (non-demo) login: requires a reachable `qknee.api.auth`
    backend. Deliberately does NOT fall back to accepting an arbitrary
    password locally when the API is unreachable — that would silently
    "authenticate" any credentials against nothing, a real security
    footgun, not a harmless demo convenience. Only the dedicated Demo
    Account path (`_attempt_demo_login`) gets an offline fallback."""
    api_url = _resolve_api_url()
    if not (api_url and _api_is_reachable(api_url)):
        raise _AuthServiceError(
            "The authentication service is currently unreachable. "
            "Try ⚡ 'Sign in with Demo Account' below for offline access, or try again shortly."
        )
    return _login_via_api(api_url, username, password)


def _signup_via_api_or_error(payload: dict) -> dict:
    api_url = _resolve_api_url()
    if not (api_url and _api_is_reachable(api_url)):
        raise _AuthServiceError(
            "The registration service is currently unreachable. "
            "Try ⚡ 'Sign in with Demo Account' below for offline access, or try again shortly."
        )
    return _signup_via_api(api_url, payload)


def _local_demo_session_payload() -> dict:
    """A clearly-labeled, offline-only guest session — used only when
    `_attempt_demo_login` finds no reachable `qknee.api.auth` backend at
    all. `access_token` is a placeholder string, never a real JWT, and is
    never sent to any HTTP endpoint (in this mode there is none to send it
    to) — this is a UI-only "you're in demo mode" session, mirroring
    `qknee.ui.dashboard`'s existing live/mock inference fallback."""
    return {
        "access_token": "local-demo-session-not-a-real-jwt",
        "token_type": "bearer",
        "user": {
            "id": "local-demo",
            "username": DEMO_USERNAME,
            "role": DEMO_ROLE,
            "full_name": DEMO_FULL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


# --------------------------------------------------------------------------- #
# Login-attempt throttling (client-side, per browser session — a real
# deployment should ALSO rate-limit at the API layer; this only protects
# this one Streamlit session's UI from rapid-fire resubmission, it is not
# a substitute for server-side throttling)
# --------------------------------------------------------------------------- #

def _register_failed_attempt() -> None:
    attempts = st.session_state.get(_FAILED_ATTEMPTS_KEY, 0) + 1
    st.session_state[_FAILED_ATTEMPTS_KEY] = attempts
    if attempts >= MAX_LOGIN_ATTEMPTS:
        st.session_state[_LOCKOUT_UNTIL_KEY] = time.time() + LOCKOUT_SECONDS


def _reset_failed_attempts() -> None:
    st.session_state[_FAILED_ATTEMPTS_KEY] = 0
    st.session_state.pop(_LOCKOUT_UNTIL_KEY, None)


def _lockout_remaining_seconds() -> float:
    until = st.session_state.get(_LOCKOUT_UNTIL_KEY)
    if until is None:
        return 0.0
    return max(0.0, until - time.time())


# --------------------------------------------------------------------------- #
# Session application + post-auth navigation
# --------------------------------------------------------------------------- #

def _apply_authenticated_session(token_payload: dict) -> None:
    user = token_payload.get("user", {}) or {}
    st.session_state[AUTHENTICATED_KEY] = True
    st.session_state[TOKEN_KEY] = token_payload.get("access_token", "")
    st.session_state[USER_INFO_KEY] = {
        "username": user.get("username", "unknown"),
        "role": user.get("role", DEMO_ROLE),
        "full_name": user.get("full_name") or user.get("username", "User"),
        "affiliation": user.get("affiliation"),
        "email": user.get("email"),
        "created_at": user.get("created_at"),
    }


def _navigate_after_auth() -> None:
    """Sends a freshly authenticated session to whatever the visitor was
    trying to reach — the workspace tab a landing-page CTA or the
    "Benchmarks"/"Switch to Diagnostic Workspace" nav buttons had already
    requested (`landing_page.VIEW_STATE_KEY`), or the Diagnostic tab by
    default when there was no pending destination (e.g. they used the
    plain top-nav "Sign In" button)."""
    from qknee.ui import landing_page

    requested_view = st.session_state.get(landing_page.VIEW_STATE_KEY, landing_page.VIEW_LANDING)
    if requested_view == landing_page.VIEW_LANDING:
        st.session_state[landing_page.VIEW_STATE_KEY] = landing_page.VIEW_DIAGNOSTIC
    st.session_state[CURRENT_PAGE_KEY] = PAGE_WORKSPACE


def _go_to_landing() -> None:
    from qknee.ui import landing_page

    st.session_state[landing_page.VIEW_STATE_KEY] = landing_page.VIEW_LANDING
    st.session_state[CURRENT_PAGE_KEY] = PAGE_LANDING
    st.rerun()


def _go_to_workspace_tab(view: str) -> None:
    """Routes straight into the workspace requesting tab `view`
    (`landing_page.VIEW_DIAGNOSTIC`/`VIEW_BENCHMARK`) — reachable while
    logged out too (the Benchmarks tab is static precomputed data, no
    live inference, so it doesn't need auth); `dashboard.render_diagnostic_tab`
    is what actually enforces the Diagnostic tab's own auth gate."""
    from qknee.ui import landing_page

    st.session_state[landing_page.VIEW_STATE_KEY] = view
    st.session_state[CURRENT_PAGE_KEY] = PAGE_WORKSPACE
    st.rerun()


def _log_out() -> None:
    from qknee.ui import landing_page

    for key in (AUTHENTICATED_KEY, TOKEN_KEY, USER_INFO_KEY):
        st.session_state.pop(key, None)
    _reset_failed_attempts()
    st.session_state[landing_page.VIEW_STATE_KEY] = landing_page.VIEW_LANDING
    st.session_state[CURRENT_PAGE_KEY] = PAGE_LANDING
    st.rerun()


# --------------------------------------------------------------------------- #
# Login tab
# --------------------------------------------------------------------------- #

def _attempt_login(username: str, password: str) -> None:
    try:
        token_payload = _login_via_api_or_error(username, password)
    except _InvalidCredentialsError as exc:
        _register_failed_attempt()
        st.error(str(exc))
        return
    except _AuthServiceError as exc:
        st.error(str(exc))
        return

    _apply_authenticated_session(token_payload)
    _reset_failed_attempts()
    _navigate_after_auth()
    st.rerun()


def _attempt_demo_login() -> None:
    api_url = _resolve_api_url()
    if api_url and _api_is_reachable(api_url):
        try:
            token_payload = _login_via_api(api_url, DEMO_USERNAME, DEMO_PASSWORD)
        except _InvalidCredentialsError:
            # First "Sign in with Demo Account" click on a fresh
            # deployment: the shared demo account doesn't exist in
            # qknee/api/users.json yet — provision it once, then log in.
            # Every later click (from anyone) reuses the same account.
            try:
                _signup_via_api(api_url, {"username": DEMO_USERNAME, "password": DEMO_PASSWORD, "role": DEMO_ROLE})
                token_payload = _login_via_api(api_url, DEMO_USERNAME, DEMO_PASSWORD)
            except _AuthServiceError as exc:
                st.error(f"Demo sign-in failed: {exc}")
                return
        except _AuthServiceError as exc:
            st.error(f"Demo sign-in failed: {exc}")
            return
    else:
        token_payload = _local_demo_session_payload()

    token_payload.setdefault("user", {}).setdefault("full_name", DEMO_FULL_NAME)
    _apply_authenticated_session(token_payload)
    _reset_failed_attempts()
    _navigate_after_auth()
    st.rerun()


def render_login_tab() -> None:
    remaining = _lockout_remaining_seconds()
    locked = remaining > 0

    if locked:
        st.error(f"Too many failed sign-in attempts. Try again in {int(remaining) + 1}s.")

    with st.form("qknee_login_form", clear_on_submit=False):
        username = st.text_input("Email or Username", key="qknee_login_username")
        password = st.text_input("Password", type="password", key="qknee_login_password")
        submitted = st.form_submit_button(
            "Sign In", type="primary", disabled=locked, use_container_width=True,
        )

    if submitted and not locked:
        if not username or not password:
            st.error("Enter both a username/email and password.")
        else:
            _attempt_login(username, password)

    st.divider()
    st.caption("Just exploring? Skip the form entirely:")
    if st.button("Sign in with Demo Account", key="qknee_demo_login", use_container_width=True, disabled=locked):
        _attempt_demo_login()


# --------------------------------------------------------------------------- #
# Signup tab
# --------------------------------------------------------------------------- #

def _derive_username_from_email(email: str) -> str:
    """`qknee.api.auth.UserCreate.username` requires 3-64 chars matching
    `^[a-zA-Z0-9_.-]+$` — this UI only asks for an email, so the username
    the backend actually stores is derived transparently from its local
    part (the ROLE/affiliation/full-name fields aren't part of the
    backend's `UserCreate` schema yet either; see `_attempt_signup`'s
    docstring)."""
    local_part = email.split("@", 1)[0]
    sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "", local_part)
    if len(sanitized) < 3:
        sanitized = (sanitized + "user")[:64]
    return sanitized[:64]


def _validate_signup_fields(full_name: str, email: str, password: str) -> list[str]:
    errors = []
    if not full_name.strip():
        errors.append("Full name is required.")
    if "@" not in email or "." not in email.split("@")[-1]:
        errors.append("Enter a valid email address.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    return errors


def _attempt_signup(full_name: str, affiliation: str, email: str, password: str, role_label: str) -> None:
    """Registers via `qknee.api.auth`'s real `/signup` + `/login` endpoints,
    then immediately authenticates the session ("Create Account" behaves
    like instant sign-in, not a separate two-step flow).

    NOTE: `full_name`/`affiliation`/`email` are collected here for a
    complete signup UX, but `qknee.api.auth.UserCreate`/`UserResponse`
    don't persist them yet (only `username`/`password`/`role`) — they're
    stashed client-side into `user_info` for this session's own display
    (nav badge, workspace header) rather than silently dropped. A
    production deployment should extend the backend schema to actually
    persist them.
    """
    backend_role = UI_ROLE_TO_BACKEND_ROLE[role_label]
    username = _derive_username_from_email(email)

    try:
        _signup_via_api_or_error({"username": username, "password": password, "role": backend_role})
        token_payload = _login_via_api_or_error(username, password)
    except _UsernameTakenError:
        st.error(f"An account already exists for '{email}'. Try signing in instead.")
        return
    except _AuthServiceError as exc:
        st.error(f"Registration failed: {exc}")
        return

    token_payload.setdefault("user", {})
    token_payload["user"]["full_name"] = full_name
    token_payload["user"]["affiliation"] = affiliation
    token_payload["user"]["email"] = email

    _apply_authenticated_session(token_payload)
    st.success(f"Welcome, {full_name}! Your account has been created.")
    _navigate_after_auth()
    st.rerun()


def render_signup_tab() -> None:
    with st.form("qknee_signup_form", clear_on_submit=False):
        full_name = st.text_input("Full Name", key="qknee_signup_full_name")
        affiliation = st.text_input("Hospital / Clinic Affiliation", key="qknee_signup_affiliation")
        email = st.text_input("Email", key="qknee_signup_email")
        password = st.text_input(
            "Password", type="password", key="qknee_signup_password", help="At least 8 characters.",
        )
        role_label = st.selectbox("Role", UI_ROLES, key="qknee_signup_role")
        submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

    if submitted:
        errors = _validate_signup_fields(full_name, email, password)
        if errors:
            for error in errors:
                st.error(error)
        else:
            _attempt_signup(full_name, affiliation, email, password, role_label)


# --------------------------------------------------------------------------- #
# Combined login/signup page
# --------------------------------------------------------------------------- #

def render_auth_page(default_tab: str = PAGE_LOGIN) -> None:
    """Renders the Sign In / Create Account tabs. `default_tab` controls
    which tab is active first (`PAGE_SIGNUP` when the visitor arrived via
    "Get Started", `PAGE_LOGIN` otherwise) — via the same tab-label-order
    trick `qknee.ui.dashboard.main()` uses, since `st.tabs()` has no
    native "start on tab N" API."""
    st.markdown("## Welcome to Q-Knee")
    st.caption("Sign in to access the live diagnostic workspace, or create a free account.")

    login_label, signup_label = "Sign In", "Create Account"
    tab_labels = [signup_label, login_label] if default_tab == PAGE_SIGNUP else [login_label, signup_label]
    tabs_by_label = dict(zip(tab_labels, st.tabs(tab_labels)))

    with tabs_by_label[login_label]:
        render_login_tab()
    with tabs_by_label[signup_label]:
        render_signup_tab()

    st.divider()
    if st.button("← Back to Home", key="qknee_auth_back_home"):
        _go_to_landing()


# --------------------------------------------------------------------------- #
# Persistent top navigation toolbar
# --------------------------------------------------------------------------- #

# Center pill-nav destinations. "Clinical Audit" has no dedicated feature
# yet, so it routes to the authenticated workspace gate (sign-in / account)
# rather than inventing a page — the closest real destination for an
# audit-trail-style view in this product today.
_NAV_ITEM_WORKSTATION = "Workstation"
_NAV_ITEM_BENCHMARKS = "Performance Benchmarks"
_NAV_ITEM_AUDIT = "Clinical Audit"


def _initials(name: str) -> str:
    parts = [part for part in name.strip().split() if part]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _active_nav_item() -> Optional[str]:
    from qknee.ui import landing_page

    current_page = st.session_state.get(CURRENT_PAGE_KEY, PAGE_LANDING)
    if current_page in (PAGE_LOGIN, PAGE_SIGNUP):
        return _NAV_ITEM_AUDIT
    if current_page != PAGE_WORKSPACE:
        return None
    requested_view = st.session_state.get(landing_page.VIEW_STATE_KEY, landing_page.VIEW_DIAGNOSTIC)
    return _NAV_ITEM_BENCHMARKS if requested_view == landing_page.VIEW_BENCHMARK else _NAV_ITEM_WORKSTATION


def render_global_navbar() -> None:
    """Single, centralized top navigation bar — brand mark on the left, a
    pill-style segmented control (Workstation / Performance Benchmarks /
    Clinical Audit) centered, and a system-status badge plus a Clinician
    Portal / Sign In (or account) control on the right. Call this exactly
    once, near the top of `dashboard.main()`, before any page body
    renders — it does not re-render on tab switches within a page."""
    from qknee.ui import landing_page

    active = _active_nav_item()

    # Unequal column weights, not an equal 3-way split: "Performance
    # Benchmarks" is roughly twice as many characters as "Workstation" —
    # forcing all three pills to the same width is what was clipping it.
    # `status_col` needs enough of its own room for both the status pill
    # and "Clinician Portal" side by side — too little here is what made
    # it visually overlap the last nav pill.
    brand_col, nav_col, status_col = st.columns([1.0, 2.6, 2.1])

    with brand_col:
        st.markdown(
            f'<div style="display:flex; align-items:center; height:2.2rem;">'
            f'<span class="qknee-brand-mark">Q</span>'
            f'<div><div style="font-weight:800; font-size:0.95rem; color:{theme.STERILE_WHITE}; '
            f'letter-spacing:-0.01em; text-transform:uppercase;">Q-Knee Clinical Workstation</div>'
            f'<div style="font-size:0.66rem; color:{theme.TEXT_MUTED}; letter-spacing:0.04em;">'
            f'Orthopedic MRI Research Division</div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with nav_col:
        # `key="qknee_navbar"` gives this container a real, stable wrapper
        # div (Streamlit's `.st-key-qknee_navbar` class) that genuinely
        # nests everything rendered inside it — unlike separate
        # `st.markdown` calls, this is how `theme.inject_clinical_theme()`'s
        # `.st-key-qknee_navbar .stButton > button` rule actually reaches
        # only these three pills (active-tab slate highlight, min-width
        # against clipping) without also restyling every other button.
        with st.container(key="qknee_navbar"):
            pill_cols = st.columns([1.0, 1.7, 1.2])
            nav_items = [
                (_NAV_ITEM_WORKSTATION, landing_page.VIEW_DIAGNOSTIC),
                (_NAV_ITEM_BENCHMARKS, landing_page.VIEW_BENCHMARK),
                (_NAV_ITEM_AUDIT, None),
            ]
            for pill_col, (label, view) in zip(pill_cols, nav_items):
                with pill_col:
                    button_type = "primary" if label == active else "secondary"
                    if st.button(label, key=f"qknee_nav_pill_{label}", use_container_width=True, type=button_type):
                        if view is not None:
                            _go_to_workspace_tab(view)
                        else:
                            st.session_state[CURRENT_PAGE_KEY] = PAGE_LOGIN
                            st.rerun()

    with status_col:
        right_cols = st.columns([1.3, 1.3])
        with right_cols[0]:
            st.markdown(
                '<div style="display:flex; align-items:center; height:2.2rem; justify-content:flex-end;">'
                '<span class="qknee-status-pill"><span class="qknee-status-dot"></span>System Operational</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        with right_cols[1]:
            if st.session_state.get(AUTHENTICATED_KEY, False):
                user_info = st.session_state.get(USER_INFO_KEY) or {}
                display_name = user_info.get("full_name") or user_info.get("username", "User")
                if st.button(_initials(display_name), key="qknee_nav_account", use_container_width=True,
                             help=f"{display_name} — Log Out"):
                    _log_out()
            else:
                if st.button("Clinician Portal", key="qknee_nav_signin", type="primary", use_container_width=True):
                    st.session_state[CURRENT_PAGE_KEY] = PAGE_LOGIN
                    st.rerun()

    st.markdown("<hr style='margin: 0.4rem 0 1rem 0;'>", unsafe_allow_html=True)


def render_top_nav() -> None:
    """Backwards-compatible alias for `render_global_navbar` — the single
    top-level frame component `dashboard.main()` calls once."""
    render_global_navbar()


def is_authenticated() -> bool:
    """Convenience predicate for callers (e.g.
    `dashboard.render_diagnostic_tab`'s auth gate) that just need a
    boolean, without importing the raw session-state key name."""
    return bool(st.session_state.get(AUTHENTICATED_KEY, False))


def can_run_inference() -> bool:
    """Whether the current session's role is allowed to run live
    diagnostic inference — `radiologist`/`triage_nurse` (Radiologist /
    Clinical Researcher), matching `qknee.api.auth.INFERENCE_ROLES`.
    `guest_demo` (Student Evaluator) is read-only: they get the landing
    page's free sample showcase, not live upload+inference. Returns
    `False` if not authenticated at all."""
    if not is_authenticated():
        return False
    user_info = st.session_state.get(USER_INFO_KEY) or {}
    return user_info.get("role") in _CLINICAL_INFERENCE_ROLES
