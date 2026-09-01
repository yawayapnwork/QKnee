"""
Q-Knee shared clinical design system.

Centralizes the visual identity, copy constants, and small rendering
helpers that `qknee.ui.landing_page`, `qknee.ui.analysis_app`, and
`qknee.ui.dashboard` all build on, so the public landing page, the
PACS-style diagnostic workstation, and the clinical dashboard read as one
coherent product rather than three independently themed pages.

Palette: deep diagnostic carbon (`#0A0E17`) page background with seamless
dark surfaces (`#111827`) for cards/masthead/sidebar, crisp slate
dividers (`#1F2937` / `#374151`), a pure-white header/high-legibility-
muted-body/sharp-micro-copy text hierarchy (`#F9FAFB` / `#9CA3AF` /
`#6B7280`), precision surgical-cyan accents (`#06B6D4` / `#0EA5E9`) for
active/interactive state, clinical status green (`#10B981`) for
low-risk/system-nominal indicators, and a dedicated primary-action blue
(`#0284C7`) for the Clinician Portal CTA. Typography is Inter (falling
back to SF Pro Display / system sans) for copy and JetBrains Mono for
tabular metrics, no decorative emoji or cartoonish iconography — status
and risk indicators are conveyed through formal badges instead.

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import cv2
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens — dark clinical diagnostic theme
# --------------------------------------------------------------------------- #

APP_BACKGROUND = "#0A0E17"       # deep diagnostic carbon — page background
CARD_SURFACE = "#111827"         # seamless dark surface — cards, masthead, sidebar panels
SURFACE_IVORY = "#111827"        # kept for call-site compat — dark theme has no separate "ivory" tier
STERILE_WHITE = "#F9FAFB"        # pure-white header/primary-copy color (kept name for call-site compat)
FOREST_GREEN = "#06B6D4"         # legacy alias, now the surgical-cyan accent (chart "primary accent"
                                 # slots in analysis_app.py/dashboard.py — headings use STERILE_WHITE
                                 # directly, not this alias; see inject_clinical_theme()'s h1-h6 rule).
SAGE_GREEN = "#10B981"           # clinical status green — low-risk fills, success/online indicators
SAGE_TINT = "#10B98122"          # translucent green chip background
DIAGNOSTIC_BLUE = "#06B6D4"      # surgical cyan — primary interactive/active-state accent
CYAN_SECONDARY = "#0EA5E9"       # secondary cyan — gradients/hover variants alongside DIAGNOSTIC_BLUE
PRIMARY_ACTION_BLUE = "#0284C7"  # Clinician Portal / primary CTA fill — distinct from the cyan accent
BORDER_GREY = "#1F2937"          # 1px card/divider borders
BORDER_STRONG = "#374151"        # emphasized divider / hover border
TEXT_MUTED = "#9CA3AF"           # high-legibility muted body copy
TEXT_FAINT = "#6B7280"           # sharp technical micro-copy
AMBER = "#F59E0B"                # dark amber — regulatory disclosure banner, moderate-risk badge

# Kept for backward-compat with call sites still referencing the old
# token names from an earlier light-theme pass; both now resolve into the
# dark palette above.
CLINICAL_SLATE = APP_BACKGROUND
SURGICAL_TEAL = SAGE_GREEN

RADIUS_SHARP = "8px"
CARD_SHADOW = "0 4px 16px rgba(0, 0, 0, 0.35)"

RISK_LOW = "#10B981"
RISK_MODERATE = "#F59E0B"
RISK_HIGH = "#EF4444"

# Recognized medical/clinical symbol (Unicode "Staff of Aesculapius"),
# not a decorative emoji — used as the sole browser-tab glyph across all
# three modules so the tab icon stays consistent with the "no playful
# emoji" mandate.
CLINICAL_GLYPH = "⚕"

MODEL_VERSION = "v1.0.4-NISQ"
SYSTEM_STATUS_LABEL = f"System Online • Release {MODEL_VERSION}"

INSTITUTION_NAME = "Q-Knee Diagnostic Imaging Laboratory"
INSTITUTION_DIVISION = "Quantum-Assisted Musculoskeletal Radiology Research Division"

DISCLOSURE_BANNER_TEXT = (
    "<b>NISQ Clinical Research Disclosure.</b> Q-Knee is a noise-intermediate-scale-quantum "
    "(NISQ) research prototype for orthopedic MRI triage. It has not been cleared or approved "
    "by any regulatory body and is not validated for standalone diagnostic use. All output "
    "requires confirmatory interpretation by a licensed radiologist."
)

NOT_A_DEVICE_FOOTNOTE = (
    "Research prototype — not a certified medical device. Not for clinical use."
)


# --------------------------------------------------------------------------- #
# Global CSS injection
# --------------------------------------------------------------------------- #

def inject_clinical_theme() -> None:
    """Injects the shared clinical design system once per script rerun —
    palette, Inter/JetBrains-Mono typography, and the formal section-card/
    badge/masthead primitives every `render_*` function in `qknee.ui`
    builds on. Idempotent (plain CSS, safe to call on every rerun)."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stApp, .stMarkdown, .stButton, .stMetric {{
            font-family: 'Inter', -apple-system, 'SF Pro Display', 'Segoe UI', sans-serif !important;
        }}
        .stApp {{
            background-color: {APP_BACKGROUND};
        }}
        /* Real layout-width constraint: Streamlit renders every st.markdown
        call into its own isolated DOM fragment, so an opening <div> tag in
        one call and a closing one in a later call never actually nest the
        widgets rendered in between — this rule against Streamlit's real
        block-container element is what actually centers/caps the page
        width; the literal `.clinical-container` wrapper divs are kept
        alongside it for markup clarity, not because they nest anything. */
        [data-testid="stAppViewContainer"] .main .block-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding-top: 1.5rem;
        }}
        .clinical-container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        [data-testid="stSidebar"] {{
            background-color: {CARD_SURFACE};
            border-right: 1px solid {BORDER_GREY};
        }}
        [data-testid="stSidebar"] * {{
            color: {STERILE_WHITE};
        }}
        [data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: {TEXT_MUTED} !important;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 700 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.02em !important;
            color: {STERILE_WHITE} !important;
        }}
        p, span, label, div {{
            line-height: 1.45;
            color: {STERILE_WHITE};
        }}
        .stCaption, [data-testid="stCaptionContainer"] {{
            color: {TEXT_MUTED} !important;
        }}

        /* Tabular monospace for all metric/numeric readouts */
        [data-testid="stMetricValue"], [data-testid="stMetricDelta"], .qknee-mono {{
            font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace !important;
            font-variant-numeric: tabular-nums;
            color: {DIAGNOSTIC_BLUE} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED} !important;
            text-transform: uppercase;
            font-size: 0.72rem !important;
            letter-spacing: 0.04em;
        }}

        /* Institutional masthead (standalone analysis_app.py entry point) */
        .qknee-masthead {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.6rem;
            padding: 0.85rem 1.2rem;
            background: {CARD_SURFACE};
            border: 1px solid {BORDER_GREY};
            border-radius: {RADIUS_SHARP};
            box-shadow: {CARD_SHADOW};
            margin-bottom: 0.7rem;
        }}
        .qknee-masthead-brand {{
            font-size: 1.0rem;
            font-weight: 800;
            color: {STERILE_WHITE};
            letter-spacing: -0.01em;
            text-transform: uppercase;
        }}
        .qknee-masthead-division {{
            font-size: 0.7rem;
            color: {TEXT_MUTED};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.1rem;
        }}
        .qknee-brand-mark {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.55rem;
            height: 1.55rem;
            border-radius: {RADIUS_SHARP};
            background: linear-gradient(135deg, {DIAGNOSTIC_BLUE} 0%, {CYAN_SECONDARY} 100%);
            color: {APP_BACKGROUND};
            font-weight: 800;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }}

        /* System-status pill: sleek border + a genuinely glowing, pulsing dot */
        .qknee-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.7rem;
            font-weight: 700;
            color: {SAGE_GREEN};
            background: {SAGE_TINT};
            border: 1px solid {SAGE_GREEN}55;
            border-radius: 999px;
            padding: 0.32rem 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }}
        .qknee-status-dot {{
            width: 0.45rem;
            height: 0.45rem;
            border-radius: 50%;
            background: {SAGE_GREEN};
            box-shadow: 0 0 0 0 {SAGE_GREEN}88;
            animation: qknee-pulse 2s infinite;
        }}
        @keyframes qknee-pulse {{
            0%   {{ box-shadow: 0 0 0 0 {SAGE_GREEN}70; }}
            70%  {{ box-shadow: 0 0 0 6px {SAGE_GREEN}00; }}
            100% {{ box-shadow: 0 0 0 0 {SAGE_GREEN}00; }}
        }}

        /* Regulatory disclosure — compact dark amber/slate banner, no
        collapsible expander */
        .qknee-disclosure-banner {{
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
            font-size: 0.74rem;
            line-height: 1.5;
            color: {TEXT_MUTED};
            background: {AMBER}14;
            border: 1px solid {AMBER}40;
            border-left: 3px solid {AMBER};
            border-radius: {RADIUS_SHARP};
            padding: 0.65rem 0.95rem;
            margin: 0.9rem 0 1.3rem 0;
        }}
        .qknee-disclosure-banner .qknee-disclosure-glyph {{
            color: {AMBER};
            font-weight: 700;
            flex-shrink: 0;
        }}
        .qknee-disclosure-banner b {{ color: {STERILE_WHITE}; }}

        /* Section cards (flat variant — sidebar/status panels) */
        .qknee-card {{
            background: {CARD_SURFACE};
            border: 1px solid {BORDER_GREY};
            border-radius: {RADIUS_SHARP};
            box-shadow: {CARD_SHADOW};
            padding: 1.05rem 1.15rem;
            height: 100%;
        }}
        .qknee-card-title {{
            font-size: 0.9rem;
            font-weight: 700;
            color: {STERILE_WHITE};
            margin-bottom: 0.3rem;
            letter-spacing: -0.005em;
        }}
        .qknee-card-body {{
            font-size: 0.79rem;
            color: {TEXT_MUTED};
        }}

        /* Glassmorphic capability cards (landing page technical grid) */
        .qknee-card-glass {{
            background: rgba(17, 24, 39, 0.7);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid {BORDER_GREY};
            border-radius: 8px;
            padding: 20px;
            height: 100%;
            transition: border-color 0.15s ease;
        }}
        .qknee-card-glass:hover {{
            border-color: {BORDER_STRONG};
        }}
        .qknee-card-pill {{
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: {DIAGNOSTIC_BLUE};
            background: {DIAGNOSTIC_BLUE}1A;
            border: 1px solid {DIAGNOSTIC_BLUE}44;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            margin-bottom: 0.7rem;
        }}
        .qknee-card-metric {{
            display: inline-block;
            font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
            font-size: 0.72rem;
            font-weight: 600;
            color: {SAGE_GREEN};
            background: {SAGE_TINT};
            border-radius: 4px;
            padding: 0.15rem 0.45rem;
            margin-top: 0.7rem;
            margin-right: 0.35rem;
        }}

        .qknee-eyebrow {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: {DIAGNOSTIC_BLUE};
            margin-bottom: 0.35rem;
        }}

        /* Risk / status badges (formal, no emoji) */
        .qknee-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            border-radius: 999px;
            padding: 0.24rem 0.6rem;
        }}
        .qknee-badge-low {{ color: {RISK_LOW}; background: {RISK_LOW}1F; border: 1px solid {RISK_LOW}55; }}
        .qknee-badge-moderate {{ color: {RISK_MODERATE}; background: {RISK_MODERATE}1F; border: 1px solid {RISK_MODERATE}55; }}
        .qknee-badge-high {{ color: {RISK_HIGH}; background: {RISK_HIGH}1F; border: 1px solid {RISK_HIGH}55; }}
        .qknee-badge-info {{ color: {DIAGNOSTIC_BLUE}; background: {DIAGNOSTIC_BLUE}1A; border: 1px solid {DIAGNOSTIC_BLUE}55; }}
        .qknee-badge-neutral {{ color: {TEXT_MUTED}; background: {TEXT_FAINT}1A; border: 1px solid {BORDER_GREY}; }}

        .qknee-ci {{
            font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            margin-top: 0.15rem;
        }}

        /* Buttons: crisp, dark-theme, cyan-accented */
        .stButton > button, .stDownloadButton > button {{
            border-radius: {RADIUS_SHARP};
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid {BORDER_GREY};
            background-color: {CARD_SURFACE};
            color: {STERILE_WHITE};
            white-space: nowrap;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: {DIAGNOSTIC_BLUE};
            color: {DIAGNOSTIC_BLUE};
            background-color: #16213366;
        }}
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
            background-color: {PRIMARY_ACTION_BLUE};
            border-color: {PRIMARY_ACTION_BLUE};
            color: {STERILE_WHITE};
        }}
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{
            background-color: #0369A1;
            border-color: #0369A1;
            color: {STERILE_WHITE};
        }}

        /* Form widgets: selects, uploaders, text inputs, dataframes */
        [data-testid="stFileUploaderDropzone"], .stSelectbox > div > div, .stTextInput > div > div,
        .stNumberInput > div > div, [data-baseweb="select"] > div {{
            background-color: {CARD_SURFACE} !important;
            border-color: {BORDER_GREY} !important;
            color: {STERILE_WHITE} !important;
        }}
        [data-testid="stDataFrame"], [data-testid="stTable"] {{
            border: 1px solid {BORDER_GREY};
            border-radius: {RADIUS_SHARP};
        }}
        .stProgress > div > div > div > div {{
            background-color: {DIAGNOSTIC_BLUE};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            border-bottom: 1px solid {BORDER_GREY};
        }}
        .stTabs [aria-selected="true"] {{
            color: {DIAGNOSTIC_BLUE} !important;
            border-bottom-color: {DIAGNOSTIC_BLUE} !important;
        }}

        hr {{ border-color: {BORDER_GREY}; }}

        /* Integrated tab-bar nav pills (qknee.ui.auth_view.render_global_navbar) —
        scoped to `st.container(key="qknee_navbar")`'s real generated
        wrapper class (Streamlit actually nests everything rendered inside
        a keyed container, unlike separate st.markdown calls, so this
        selector genuinely reaches only the nav pills). Active/inactive is
        Streamlit's own `type="primary"`/`type="secondary"`. `white-space:
        nowrap` plus a `min-width` sized to the longest label — not
        `nowrap` alone — is what actually stops "Performance Benchmarks"
        from clipping: nowrap with no room to grow just clips harder. */
        .st-key-qknee_navbar .stButton > button {{
            width: 100%;
            white-space: nowrap;
            overflow: visible;
            padding: 0.5rem 0.85rem;
            font-size: 0.8rem;
        }}
        .st-key-qknee_navbar .stButton > button[kind="secondary"] {{
            background-color: transparent;
            border: 1px solid transparent;
            color: {TEXT_MUTED};
        }}
        .st-key-qknee_navbar .stButton > button[kind="secondary"]:hover {{
            background-color: {CARD_SURFACE};
            border-color: {BORDER_GREY};
            color: {STERILE_WHITE};
        }}
        .st-key-qknee_navbar .stButton > button[kind="primary"] {{
            background-color: #1E293B;
            border: 1px solid #1E293B;
            border-bottom: 2px solid {DIAGNOSTIC_BLUE};
            color: #FFFFFF;
            border-radius: {RADIUS_SHARP} {RADIUS_SHARP} 0 0;
        }}
        .st-key-qknee_navbar .stButton > button[kind="primary"]:hover {{
            background-color: #243449;
            border-color: #243449;
            border-bottom: 2px solid {DIAGNOSTIC_BLUE};
            color: #FFFFFF;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Institutional masthead + disclosure banner
# --------------------------------------------------------------------------- #

def render_institutional_masthead(active_module: str) -> None:
    """Renders the institutional header shared by every page: laboratory
    branding, current module name, and the live system-status indicator
    (`System Online • Model v1.0.4-NISQ`)."""
    st.markdown(
        f"""
        <div class="qknee-masthead">
            <div style="display:flex; align-items:center;">
                <span class="qknee-brand-mark">Q</span>
                <div>
                    <div class="qknee-masthead-brand">{INSTITUTION_NAME}</div>
                    <div class="qknee-masthead-division">{INSTITUTION_DIVISION} &middot; {active_module}</div>
                </div>
            </div>
            <div class="qknee-status-pill"><span class="qknee-status-dot"></span>{SYSTEM_STATUS_LABEL}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclosure_banner() -> None:
    """Formal NISQ regulatory disclosure — a sleek, compact dark amber/
    slate banner (not a collapsible expander): a single flush-left accent
    bar, a small glyph, and 12px muted-slate copy. Called once, directly
    under the hero block, by `qknee.ui.landing_page.render_hero()`."""
    st.markdown(
        f"""
        <div class="qknee-disclosure-banner">
            <span class="qknee-disclosure-glyph">&#9888;</span>
            <span>{DISCLOSURE_BANNER_TEXT}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Risk badges + approximate confidence intervals
# --------------------------------------------------------------------------- #

def risk_tier(value: float) -> Tuple[str, str]:
    """Maps a `[0, 1]` probability onto `(css_class_suffix, tier_label)`."""
    if value >= 0.66:
        return "high", "HIGH"
    if value >= 0.33:
        return "moderate", "MODERATE"
    return "low", "LOW"


def risk_badge_html(label: str, value: float) -> str:
    css_class, tier_label = risk_tier(value)
    return f'<span class="qknee-badge qknee-badge-{css_class}">{label}: {tier_label}</span>'


def approximate_confidence_interval(point_estimate: float, n_effective: float = 40.0) -> Tuple[float, float]:
    """Presentation-layer Wald-interval approximation (95%) around a
    single sigmoid-output probability, treating it as a Bernoulli rate
    estimated from `n_effective` pseudo-observations — the pipeline
    itself reports one point estimate, not a sampling distribution, so
    this is an explicitly-labeled *approximate* band for the triage
    panel, not a statistically calibrated clinical confidence interval.
    """
    p = min(max(point_estimate, 1e-6), 1 - 1e-6)
    margin = 1.96 * ((p * (1 - p)) / n_effective) ** 0.5
    return max(0.0, p - margin), min(1.0, p + margin)


def format_confidence_interval(point_estimate: float) -> str:
    low, high = approximate_confidence_interval(point_estimate)
    return f"95% CI (approx.): {low * 100:.1f}–{high * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Clinically-standard Grad-CAM color scales
# --------------------------------------------------------------------------- #

def _build_coolwarm_lut() -> np.ndarray:
    """Builds a 256-entry diverging blue-white-red ("Cool-Warm") lookup
    table as a `(256, 1, 3)` BGR uint8 array, matching the diverging scale
    radiological attribution overlays conventionally use for signed
    activation maps. OpenCV ships no built-in `COLORMAP_COOLWARM`
    constant, but `cv2.applyColorMap` accepts a custom LUT array in place
    of the enum, so this is passed through unchanged everywhere a
    built-in colormap constant would be."""
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    cool_rgb = np.array([59, 130, 246], dtype=np.float32)    # diagnostic blue
    mid_rgb = np.array([248, 250, 252], dtype=np.float32)    # sterile white
    warm_rgb = np.array([185, 28, 28], dtype=np.float32)     # clinical red

    lower = t[:, None] * 2.0
    rgb = np.where(
        lower[:, :1] <= 1.0,
        cool_rgb[None, :] + lower * (mid_rgb[None, :] - cool_rgb[None, :]),
        mid_rgb[None, :] + (lower - 1.0) * (warm_rgb[None, :] - mid_rgb[None, :]),
    )
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = rgb[:, ::-1]
    return bgr.reshape(256, 1, 3)


_COOLWARM_LUT = _build_coolwarm_lut()

# Clinically standard scales for the Grad-CAM overlay selector — Viridis
# (perceptually uniform, the radiology-literature default for attribution
# maps), Inferno (high-contrast, good for print/low-light review), and a
# diverging Cool-Warm scale for signed-activation reading.
CLINICAL_COLORMAPS: Dict[str, Union[int, np.ndarray]] = {
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Cool-Warm": _COOLWARM_LUT,
}
DEFAULT_COLORMAP_NAME = "Viridis"


# --------------------------------------------------------------------------- #
# PACS-style crosshair + slice-depth overlay
# --------------------------------------------------------------------------- #

def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


_CROSSHAIR_BGR = _hex_to_bgr(DIAGNOSTIC_BLUE)


def draw_clinical_crosshair(image: np.ndarray, gap_fraction: float = 0.12) -> np.ndarray:
    """Draws a PACS-style medical crosshair (a gapped cross centered on
    the image, diagnostic-blue) onto a copy of `image` — marks the
    anatomical center of the displayed slice, the conventional
    radiological-viewport reference point, without obscuring it (the
    center `gap_fraction` of each arm is left open).

    Args:
        image: `(H, W)` or `(H, W, 3)` uint8 array; not modified in place.
        gap_fraction: Fraction of the shorter image dimension left open
            at the crosshair's center.

    Returns: `(H, W, 3)` BGR uint8 array with the crosshair drawn on top.
    """
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
        if canvas.shape[2] == 1:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    height, width = canvas.shape[:2]
    cy, cx = height // 2, width // 2
    gap = int(min(height, width) * gap_fraction / 2)
    thickness = max(1, min(height, width) // 300)

    overlay = canvas.copy()
    cv2.line(overlay, (0, cy), (max(cx - gap, 0), cy), _CROSSHAIR_BGR, thickness, cv2.LINE_AA)
    cv2.line(overlay, (min(cx + gap, width), cy), (width, cy), _CROSSHAIR_BGR, thickness, cv2.LINE_AA)
    cv2.line(overlay, (cx, 0), (cx, max(cy - gap, 0)), _CROSSHAIR_BGR, thickness, cv2.LINE_AA)
    cv2.line(overlay, (cx, min(cy + gap, height)), (cx, height), _CROSSHAIR_BGR, thickness, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), max(2, thickness * 2), _CROSSHAIR_BGR, thickness, cv2.LINE_AA)

    return cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0)


def slice_depth_caption(plane_label: str, index: int, max_index: int, *, primary: bool = False) -> str:
    """Formal viewport caption, e.g. `"SAGITTAL — Slice Depth 12/40 [PRIMARY]"`."""
    tag = " [PRIMARY]" if primary else ""
    return f"{plane_label.upper()} — Slice Depth {index + 1}/{max_index + 1}{tag}"
