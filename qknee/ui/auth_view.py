"""
Q-Knee authentication UI (Streamlit) — sign-in/register forms, the
persistent top navigation toolbar, and the `st.session_state` contract
that wires both into `qknee.ui.dashboard.main()`.

Session-state contract (the keys `qknee.ui.dashboard` and this module
both read/write):
    authenticated (bool) - whether the current browser session has a valid
                            (or fallback-local) authenticated session.
    token (str)          - the bearer JWT from `qknee.api.auth` (or a
                            clearly-labeled placeholder in offline/local-
                            fallback mode — see `_local_demo_session_payload`).
    user_info (dict)     - {"email", "role", "full_name", "affiliation",
                            "created_at"} — never includes a password or
                            password hash.
    current_page (str)   - one of `PAGE_LANDING` / `PAGE_LOGIN` /
                            `PAGE_SIGNUP` / `PAGE_WORKSPACE`; the top-level
                            page router `dashboard.main()` reads.

    In addition to the contract above, a successful sign-in also mirrors
    the token/profile into the plain literal keys `st.session_state.auth_token`
    / `st.session_state.current_user` — the exact contract named by this
    module's spec — so both this app's own wiring (which predates that
    naming) and any external code written against the literal spec keys
    work off the same authenticated session.

Backend wiring: form submissions call the real `qknee.api.auth` endpoints
(`POST {QKNEE_API_URL}/api/v1/auth/{register,login}`) when `$QKNEE_API_URL`
is set and reachable — same env var / reachability probe convention
`qknee.ui.dashboard.resolve_api_url`/`api_is_reachable` use for inference.
When the API is unreachable, regular sign-in and registration surface a
clear error (never silently "authenticate" an arbitrary password against
nothing — see `_login_via_api_or_error`'s docstring) — only the "Sign in
with Demo Account" button falls back to a local, clearly-labeled guest
session, so the product stays demoable end to end even with no backend
running, matching the mock-fallback convention used everywhere else in
this codebase (`qknee.ui.dashboard`, `qknee.api.server`).

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

# The spec's literal top-level session-state keys — written alongside the
# contract above on every successful sign-in (see `_apply_authenticated_session`).
SPEC_TOKEN_KEY = "auth_token"
SPEC_USER_KEY = "current_user"

# Internal-only session-state keys (login throttling).
_FAILED_ATTEMPTS_KEY = "_qknee_auth_failed_attempts"
_LOCKOUT_UNTIL_KEY = "_qknee_auth_lockout_until"

MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 30.0

LOGIN_ENDPOINT = "/api/v1/auth/login"
REGISTER_ENDPOINT = "/api/v1/auth/register"

# --------------------------------------------------------------------------- #
# Role mapping: this UI's role selector labels vs. `qknee.api.auth.ROLES`'s
# three backend role values — a direct 1:1 mapping now (the old UI copy
# used two invented labels that compressed onto a different backend role
# scheme; that compression is gone).
# --------------------------------------------------------------------------- #

UI_ROLES: tuple[str, ...] = ("Radiologist", "Clinical Researcher", "Clinical Auditor")
UI_ROLE_TO_BACKEND_ROLE: Dict[str, str] = {
    "Radiologist": "radiologist",
    "Clinical Researcher": "researcher",
    "Clinical Auditor": "clinical_auditor",
}
BACKEND_ROLE_TO_UI_LABEL: Dict[str, str] = {v: k for k, v in UI_ROLE_TO_BACKEND_ROLE.items()}

# Mirrors `qknee.api.auth.INFERENCE_ROLES` (the one role the API's
# `require_role` guard permits onto `/predict`/`/explain`/`/report`) —
# duplicated as a plain literal rather than imported, since `qknee.api.auth`
# has import-time side effects (opens a DB engine/creates tables, logs the
# insecure-default-JWT-secret warning) that don't belong in the UI process
# just to read one constant.
_CLINICAL_INFERENCE_ROLES: tuple[str, ...] = ("radiologist",)

DEMO_EMAIL = "demo.researcher@qknee-demo.org"
DEMO_PASSWORD = "QKneeDemo!2026"  # noqa: S105 - intentionally public: a shared, researcher-role-only demo account
DEMO_ROLE = "researcher"
DEMO_FULL_NAME = "Demo Researcher"


# --------------------------------------------------------------------------- #
# Auth-service errors
# --------------------------------------------------------------------------- #

class _AuthServiceError(Exception):
    """Base class for a failed auth call — message is shown to the user as-is."""


class _InvalidCredentialsError(_AuthServiceError):
    def __init__(self) -> None:
        super().__init__("Incorrect email or password.")


class _EmailTakenError(_AuthServiceError):
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


def _login_via_api(api_url: str, email: str, password: str) -> dict:
    import requests

    try:
        # The backend's `UserLogin.username` field carries the email
        # address (see qknee.api.auth) — kept as `username` on the wire so
        # this stays aligned with the OAuth2 password-grant convention
        # `oauth2_scheme`'s `tokenUrl` implies.
        response = requests.post(
            f"{api_url}{LOGIN_ENDPOINT}", json={"username": email, "password": password}, timeout=5,
        )
    except requests.RequestException as exc:
        raise _AuthServiceError(f"Could not reach the authentication service ({exc}).") from exc

    if response.status_code == 401:
        raise _InvalidCredentialsError()
    if response.status_code >= 400:
        raise _AuthServiceError(_safe_error_detail(response))
    return response.json()  # {"access_token", "token_type", "expires_in_minutes", "user": {...}}


def _register_via_api(api_url: str, payload: dict) -> dict:
    import requests

    try:
        response = requests.post(f"{api_url}{REGISTER_ENDPOINT}", json=payload, timeout=5)
    except requests.RequestException as exc:
        raise _AuthServiceError(f"Could not reach the registration service ({exc}).") from exc

    if response.status_code == 409:
        raise _EmailTakenError(_safe_error_detail(response))
    if response.status_code >= 400:
        raise _AuthServiceError(_safe_error_detail(response))
    return response.json()  # UserResponse: {"id", "email", "full_name", "role", "created_at", "is_active"}


def _login_via_api_or_error(email: str, password: str) -> dict:
    """Regular (non-demo) sign-in: requires a reachable `qknee.api.auth`
    backend. Deliberately does NOT fall back to accepting an arbitrary
    password locally when the API is unreachable — that would silently
    "authenticate" any credentials against nothing, a real security
    footgun, not a harmless demo convenience. Only the dedicated Demo
    Account path (`_attempt_demo_login`) gets an offline fallback."""
    api_url = _resolve_api_url()
    if not (api_url and _api_is_reachable(api_url)):
        raise _AuthServiceError(
            "The authentication service is currently unreachable. "
            "Try 'Sign in with Demo Account' below for offline access, or try again shortly."
        )
    return _login_via_api(api_url, email, password)


def _register_via_api_or_error(payload: dict) -> dict:
    api_url = _resolve_api_url()
    if not (api_url and _api_is_reachable(api_url)):
        raise _AuthServiceError(
            "The registration service is currently unreachable. "
            "Try 'Sign in with Demo Account' below for offline access, or try again shortly."
        )
    return _register_via_api(api_url, payload)


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
            "email": DEMO_EMAIL,
            "role": DEMO_ROLE,
            "full_name": DEMO_FULL_NAME,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_active": True,
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
    access_token = token_payload.get("access_token", "")
    user_info = {
        "email": user.get("email", "unknown"),
        "role": user.get("role", DEMO_ROLE),
        "full_name": user.get("full_name") or user.get("email", "User"),
        "affiliation": user.get("affiliation"),
        "created_at": user.get("created_at"),
    }

    st.session_state[AUTHENTICATED_KEY] = True
    st.session_state[TOKEN_KEY] = access_token
    st.session_state[USER_INFO_KEY] = user_info

    # Spec's literal contract keys — mirrors the same token/profile under
    # the exact names `st.session_state.auth_token`/`st.session_state.current_user`.
    st.session_state[SPEC_TOKEN_KEY] = access_token
    st.session_state[SPEC_USER_KEY] = {
        "user_name": user_info["full_name"],
        "role": user_info["role"],
        **user_info,
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

    for key in (AUTHENTICATED_KEY, TOKEN_KEY, USER_INFO_KEY, SPEC_TOKEN_KEY, SPEC_USER_KEY):
        st.session_state.pop(key, None)
    _reset_failed_attempts()
    st.session_state[landing_page.VIEW_STATE_KEY] = landing_page.VIEW_LANDING
    st.session_state[CURRENT_PAGE_KEY] = PAGE_LANDING
    st.rerun()


# --------------------------------------------------------------------------- #
# Sign-in tab
# --------------------------------------------------------------------------- #

def _attempt_login(email: str, password: str) -> None:
    try:
        token_payload = _login_via_api_or_error(email, password)
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
            token_payload = _login_via_api(api_url, DEMO_EMAIL, DEMO_PASSWORD)
        except _InvalidCredentialsError:
            # First "Sign in with Demo Account" click on a fresh
            # deployment: the shared demo account doesn't exist in the
            # user store yet — provision it once, then log in. Every
            # later click (from anyone) reuses the same account.
            try:
                _register_via_api(
                    api_url,
                    {"email": DEMO_EMAIL, "password": DEMO_PASSWORD, "full_name": DEMO_FULL_NAME, "role": DEMO_ROLE},
                )
                token_payload = _login_via_api(api_url, DEMO_EMAIL, DEMO_PASSWORD)
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
        email = st.text_input("Email", key="qknee_login_email")
        password = st.text_input("Password", type="password", key="qknee_login_password")
        submitted = st.form_submit_button(
            "Authenticate Credentials", type="primary", disabled=locked, use_container_width=True,
        )

    if submitted and not locked:
        if not email or not password:
            st.error("Enter both an email address and password.")
        else:
            _attempt_login(email, password)

    st.divider()
    st.caption("Just exploring? Skip the form entirely:")
    if st.button("Sign in with Demo Account", key="qknee_demo_login", use_container_width=True, disabled=locked):
        _attempt_demo_login()


# --------------------------------------------------------------------------- #
# Registration tab ("Request Institutional Access")
# --------------------------------------------------------------------------- #

_FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "icloud.com", "aol.com", "proton.me"}


def _is_institutional_email(email: str) -> bool:
    """Accepts a `.edu` domain, or a domain that looks hospital/clinic/org
    affiliated (`@hospital.org`-style); rejects common free webmail
    providers outright."""
    if "@" not in email:
        return False
    domain = email.rsplit("@", 1)[-1].strip().lower()
    if not domain or "." not in domain:
        return False
    if domain in _FREE_EMAIL_DOMAINS:
        return False
    return domain.endswith(".edu") or "hospital" in domain or "clinic" in domain or domain.endswith(".org")


def _validate_register_fields(full_name: str, email: str, password: str, confirm_password: str) -> list[str]:
    errors = []
    if not full_name.strip():
        errors.append("Full name is required.")
    if "@" not in email or "." not in email.split("@")[-1]:
        errors.append("Enter a valid email address.")
    elif not _is_institutional_email(email):
        errors.append("Use your institutional email address (hospital or university domain).")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    elif not any(c.isalpha() for c in password) or not any(c.isdigit() for c in password) or password.isalnum():
        errors.append("Password must contain a letter, a digit, and a special character.")
    if password != confirm_password:
        errors.append("Passwords do not match.")
    return errors


def _attempt_register(full_name: str, email: str, password: str, role_label: str) -> None:
    """Registers via `qknee.api.auth`'s real `/register` endpoint. Unlike
    the previous "signup" flow, this does NOT auto-authenticate the new
    account — it confirms registration and hands the visitor back to the
    Sign In tab, per spec."""
    backend_role = UI_ROLE_TO_BACKEND_ROLE[role_label]

    try:
        _register_via_api_or_error({"email": email, "password": password, "full_name": full_name, "role": backend_role})
    except _EmailTakenError:
        st.error(f"An account already exists for '{email}'. Try signing in instead.")
        return
    except _AuthServiceError as exc:
        st.error(f"Registration failed: {exc}")
        return

    st.success(f"Institutional access granted for {full_name}. Sign in with your new credentials below.")
    st.session_state[CURRENT_PAGE_KEY] = PAGE_LOGIN
    st.rerun()


def render_signup_tab() -> None:
    with st.form("qknee_register_form", clear_on_submit=False):
        full_name = st.text_input("Full Name", key="qknee_register_full_name")
        email = st.text_input(
            "Institutional Email", key="qknee_register_email",
            help="A hospital or university domain, e.g. name@hospital.org or name@university.edu.",
        )
        password = st.text_input(
            "Password", type="password", key="qknee_register_password",
            help="At least 8 characters, with a letter, a digit, and a special character.",
        )
        confirm_password = st.text_input("Confirm Password", type="password", key="qknee_register_confirm_password")
        role_label = st.selectbox("Clinical Role", UI_ROLES, key="qknee_register_role")
        submitted = st.form_submit_button("Request Institutional Access", type="primary", use_container_width=True)

    if submitted:
        errors = _validate_register_fields(full_name, email, password, confirm_password)
        if errors:
            for error in errors:
                st.error(error)
        else:
            _attempt_register(full_name, email, password, role_label)


# --------------------------------------------------------------------------- #
# Combined sign-in/register page
# --------------------------------------------------------------------------- #

def render_auth_page(default_tab: str = PAGE_LOGIN) -> None:
    """Renders the Sign In / Request Institutional Access tabs. `default_tab`
    controls which tab is active first (`PAGE_SIGNUP` when the visitor
    arrived via "Get Started"/just finished registering, `PAGE_LOGIN`
    otherwise) — via the same tab-label-order trick `qknee.ui.dashboard.main()`
    uses, since `st.tabs()` has no native "start on tab N" API."""
    st.markdown("## Welcome to Q-Knee")
    st.caption("Sign in to access the live diagnostic workspace, or request institutional access.")

    login_label, signup_label = "Clinician Sign In", "Request Institutional Access"
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
_NAV_ITEM_WORKSTATION = "Diagnostic Viewer"
_NAV_ITEM_BENCHMARKS = "Cohort Analytics & ROC"
_NAV_ITEM_AUDIT = "Clinical Audit Trail"

_AMBER_BANNER_TEXT = "View-Only Mode • Sign in for diagnostic inference & clinical reporting"


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
    Portal / Sign In (or clinician badge) control on the right. Call this
    exactly once, near the top of `dashboard.main()`, before any page body
    renders — it does not re-render on tab switches within a page."""
    from qknee.ui import landing_page

    active = _active_nav_item()

    # `nav_col` needs enough room for all three pills at their real label
    # lengths ("Diagnostic Viewer" / "Cohort Analytics & ROC" / "Clinical
    # Audit Trail" are all similar length, ~17-22 characters) — too little
    # here is what let a pill's `white-space:nowrap` label visually spill
    # out of its column and overlap its neighbor. `status_col` needs its
    # own room for the telemetry strip plus the clinician badge/Portal
    # button side by side without the two colliding either.
    brand_col, nav_col, status_col = st.columns([1.1, 3.3, 2.4])

    with brand_col:
        st.markdown(
            f'<div style="display:flex; align-items:center; height:2.2rem;">'
            f'<span class="qknee-brand-mark">Q</span>'
            f'<div><div style="font-weight:800; font-size:0.9rem; color:{theme.STERILE_WHITE}; '
            f'letter-spacing:-0.005em; text-transform:uppercase;">{theme.INSTITUTION_NAME}</div>'
            f'<div style="font-size:0.62rem; color:{theme.TEXT_MUTED}; letter-spacing:0.03em;">'
            f'{theme.INSTITUTION_DIVISION}</div>'
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
            pill_cols = st.columns([1.05, 1.35, 1.2])
            nav_items = [
                (_NAV_ITEM_WORKSTATION, landing_page.VIEW_DIAGNOSTIC),
                (_NAV_ITEM_BENCHMARKS, landing_page.VIEW_BENCHMARK),
                (_NAV_ITEM_AUDIT, None),
            ]
            for pill_col, (label, view) in zip(pill_cols, nav_items):
                with pill_col:
                    label = str(label)
                    btn_type = "primary" if label == active else "secondary"
                    clean_key = f"qknee_nav_pill_{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}"
                    if st.button(label, key=clean_key, use_container_width=True, type=btn_type):
                        if view is not None:
                            _go_to_workspace_tab(view)
                        else:
                            st.session_state[CURRENT_PAGE_KEY] = PAGE_LOGIN
                            st.rerun()

    with status_col:
        right_cols = st.columns([2.6, 1.3])
        with right_cols[0]:
            st.markdown(
                f'<div style="display:flex; align-items:center; height:2.2rem; justify-content:flex-end;">'
                f'{theme.render_telemetry_pills()}'
                f"</div>",
                unsafe_allow_html=True,
            )
        with right_cols[1]:
            if st.session_state.get(AUTHENTICATED_KEY, False):
                user_info = st.session_state.get(USER_INFO_KEY) or {}
                display_name = user_info.get("full_name") or user_info.get("email", "User")
                role_label = BACKEND_ROLE_TO_UI_LABEL.get(user_info.get("role"), user_info.get("role", "User"))
                badge_text = f"Dr. {display_name} | {role_label}"
                if st.button(_initials(display_name), key="qknee_nav_account", use_container_width=True,
                             help=f"{badge_text} — Sign Out"):
                    _log_out()
            else:
                if st.button("Clinician Portal", key="qknee_nav_signin", type="primary", use_container_width=True):
                    st.session_state[CURRENT_PAGE_KEY] = PAGE_LOGIN
                    st.rerun()

    if not st.session_state.get(AUTHENTICATED_KEY, False):
        st.markdown(
            f'<div style="margin:0.4rem 0; padding:0.45rem 0.9rem; border-radius:6px; '
            f'background:#FEF3C7; border:1px solid #FCD34D; color:#92400E; '
            f'font-size:0.78rem; font-weight:600;">{_AMBER_BANNER_TEXT}</div>',
            unsafe_allow_html=True,
        )
    else:
        user_info = st.session_state.get(USER_INFO_KEY) or {}
        display_name = user_info.get("full_name") or user_info.get("email", "User")
        role_label = BACKEND_ROLE_TO_UI_LABEL.get(user_info.get("role"), user_info.get("role", "User"))
        st.markdown(
            f'<div style="margin:0.4rem 0; padding:0.35rem 0.9rem; border-radius:6px; '
            f'background:#ECFDF5; border:1px solid #6EE7B7; color:#065F46; '
            f'font-size:0.78rem; font-weight:600;">Dr. {display_name} | {role_label}</div>',
            unsafe_allow_html=True,
        )

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
    diagnostic inference — `radiologist` only, matching
    `qknee.api.auth.INFERENCE_ROLES`. `researcher`/`clinical_auditor` are
    read-only Research Observer tiers: they get the landing page's free
    sample showcase, not live upload+inference. Returns `False` if not
    authenticated at all."""
    if not is_authenticated():
        return False
    user_info = st.session_state.get(USER_INFO_KEY) or {}
    return user_info.get("role") in _CLINICAL_INFERENCE_ROLES
