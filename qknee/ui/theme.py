"""
Q-Knee shared clinical design system.

Centralizes the visual identity, copy constants, and small rendering
helpers that `qknee.ui.landing_page`, `qknee.ui.analysis_app`, and
`qknee.ui.dashboard` all build on, so the public landing page, the
PACS-style diagnostic workstation, and the clinical dashboard read as one
coherent product rather than three independently themed pages.

Palette: sterile ivory/white surfaces (`#FBFDFE`, `#F8FAFC`, `#FFFFFF`),
deep medical forest/pine green (`#064E3B`) for institutional branding and
primary headings, muted sage/eucalyptus (`#059669` / `#10B981` /
`#D1FAE5`) for clinical badges and status chips, and charcoal-slate text
(`#0F172A` primary, `#475569` metadata, `#94A3B8` secondary — never pure
black). Typography is Inter (falling back to SF Pro Display / system
sans) for copy and JetBrains Mono for tabular metrics, no decorative
emoji or cartoonish iconography — status and risk indicators are
conveyed through formal badges instead.

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import cv2
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens — light clinical theme
# --------------------------------------------------------------------------- #

APP_BACKGROUND = "#F8FAFC"       # soft clinical white — page background
CARD_SURFACE = "#FFFFFF"         # crisp white — cards, masthead, sidebar panels
SURFACE_IVORY = "#FBFDFE"        # sterile ivory — hero/section backgrounds
STERILE_WHITE = "#0F172A"        # primary copy color (charcoal slate; kept name for call-site compat)
FOREST_GREEN = "#064E3B"         # deep medical forest/pine — branding, primary headings, active state
SAGE_GREEN = "#059669"           # muted sage/eucalyptus — clinical badges, confidence tags, fills
SAGE_TINT = "#D1FAE5"            # pale sage — chip backgrounds
DIAGNOSTIC_BLUE = "#064E3B"      # legacy alias — now maps to forest green (blue accent retired)
BORDER_GREY = "#E2E8F0"          # 1px card/divider borders
TEXT_MUTED = "#475569"           # metadata / secondary copy
TEXT_FAINT = "#94A3B8"           # tertiary/secondary labels

# Kept for backward-compat with call sites still referencing the old dark
# token names; both now resolve into the light palette above.
CLINICAL_SLATE = APP_BACKGROUND
SURGICAL_TEAL = SAGE_GREEN

RADIUS_SHARP = "8px"
CARD_SHADOW = "0 1px 3px rgba(15, 23, 42, 0.06)"

RISK_LOW = "#059669"
RISK_MODERATE = "#B45309"
RISK_HIGH = "#C2410C"       # muted terracotta — not neon red

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
        [data-testid="stSidebar"] {{
            background-color: {CARD_SURFACE};
            border-right: 1px solid {BORDER_GREY};
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 700 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.02em !important;
            color: {FOREST_GREEN} !important;
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
            color: {FOREST_GREEN} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED} !important;
            text-transform: uppercase;
            font-size: 0.72rem !important;
            letter-spacing: 0.04em;
        }}

        /* Institutional masthead / global navbar */
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
            color: {FOREST_GREEN};
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
            background: {FOREST_GREEN};
            color: {CARD_SURFACE};
            font-weight: 800;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }}
        .qknee-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.7rem;
            font-weight: 700;
            color: #065F46;
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }}
        .qknee-status-dot {{
            width: 0.4rem;
            height: 0.4rem;
            border-radius: 50%;
            background: {SAGE_GREEN};
            box-shadow: 0 0 0 3px {SAGE_GREEN}22;
        }}

        /* Disclosure banner — subtle, collapsible */
        .qknee-disclosure-text {{
            font-size: 0.68rem;
            line-height: 1.5;
            color: {TEXT_MUTED};
        }}
        .qknee-disclosure-text b {{ color: {FOREST_GREEN}; }}

        /* Section cards */
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
            color: {FOREST_GREEN};
            margin-bottom: 0.3rem;
            letter-spacing: -0.005em;
        }}
        .qknee-card-body {{
            font-size: 0.79rem;
            color: {TEXT_MUTED};
        }}
        .qknee-eyebrow {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: {SAGE_GREEN};
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
        .qknee-badge-low {{ color: #065F46; background: {SAGE_TINT}; border: 1px solid #A7F3D0; }}
        .qknee-badge-moderate {{ color: {RISK_MODERATE}; background: #FFFBEB; border: 1px solid #FDE68A; }}
        .qknee-badge-high {{ color: {RISK_HIGH}; background: #FFF7ED; border: 1px solid #FED7AA; }}
        .qknee-badge-info {{ color: {FOREST_GREEN}; background: #ECFDF5; border: 1px solid #A7F3D0; }}
        .qknee-badge-neutral {{ color: {TEXT_MUTED}; background: #F1F5F9; border: 1px solid {BORDER_GREY}; }}

        .qknee-ci {{
            font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            margin-top: 0.15rem;
        }}

        /* Buttons: crisp, formal, light-theme */
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
            border-color: {FOREST_GREEN};
            color: {FOREST_GREEN};
            background-color: #F1F5F9;
        }}
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
            background-color: {FOREST_GREEN};
            border-color: {FOREST_GREEN};
            color: {CARD_SURFACE};
        }}
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{
            background-color: #053a2b;
            color: {CARD_SURFACE};
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
            background-color: {SAGE_GREEN};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            border-bottom: 1px solid {BORDER_GREY};
        }}
        .stTabs [aria-selected="true"] {{
            color: {FOREST_GREEN} !important;
            border-bottom-color: {FOREST_GREEN} !important;
        }}

        hr {{ border-color: {BORDER_GREY}; }}
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
    """Formal NISQ regulatory disclosure — a single, subtle, collapsible
    strip (11px muted slate typography) rather than a heavy full-width
    block, per the clinical design system's disclosure requirements."""
    with st.expander("NISQ Clinical Research Disclosure", expanded=False):
        st.markdown(f'<div class="qknee-disclosure-text">{DISCLOSURE_BANNER_TEXT}</div>', unsafe_allow_html=True)


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
