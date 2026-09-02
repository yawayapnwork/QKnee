"""
Q-Knee shared clinical design system.

Centralizes the visual identity, copy constants, and small rendering
helpers that `qknee.ui.landing_page`, `qknee.ui.analysis_app`, and
`qknee.ui.dashboard` all build on, so the public landing page, the
PACS-style diagnostic workstation, and the clinical dashboard read as one
coherent, sterile institutional healthcare product rather than three
independently themed pages.

Palette: sterile surgical off-white (`#F8FAFC`) page background with pure
clinical white (`#FFFFFF`) surfaces (cards/masthead/sidebar), thin rigid
neutral borders (`#E2E8F0` / `#CBD5E1`), zero drop-shadow blur, sharp 4px
radii, an institutional midnight-slate (`#0F172A`) / muted clinical grey
(`#475569`) text hierarchy, tabular monospace (`SF Mono` / `JetBrains
Mono` / `Consolas`) for DICOM tags, coordinates, and latencies, a single
diagnostic-blue accent (`#0284C7`) for active/interactive state, clinical
green (`#16A34A`) for normal/negative findings, objective diagnostic
crimson (`#DC2626`) for high-risk/positive findings, and muted amber
(`#D97706`) for warnings/regulatory advisories. Typography is Inter
(falling back to SF Pro Display / system sans) for copy, no gradients, no
pill-shaped buttons, no decorative emoji or playful transition
animations — status and risk indicators are conveyed through square,
high-contrast badges instead.

INVESTIGATIONAL DEVICE ONLY — not cleared for primary diagnostic
determination. For adjunctive triage use only.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import cv2
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens — sterile clinical light theme
# --------------------------------------------------------------------------- #

APP_BACKGROUND = "#FFFFFF"       # clean white — page canvas (mint-tinted sections are applied
                                  # per-block via SECTION_MINT_BG, not globally)
CARD_SURFACE = "#FFFFFF"         # crisp white — cards, masthead, sidebar panels
SURFACE_IVORY = "#FFFFFF"        # kept for call-site compat — same white surface
STERILE_WHITE = "#1F2937"        # dark slate — primary text/header color (name kept for
                                  # call-site compat with earlier theme passes)
FOREST_GREEN = "#107050"         # deep surgical emerald — chart "positive"/primary accent slots
                                  # in analysis_app.py/dashboard.py (name kept for call-site compat)
SAGE_GREEN = "#16A34A"           # clinical green — low-risk fills, success/online indicators
SAGE_TINT = "#16A34A1A"          # translucent green chip background
DIAGNOSTIC_BLUE = "#56B8A0"      # soft seafoam mint — active/interactive accent, tab underlines
CYAN_SECONDARY = "#48A992"       # deeper mint — hover/pressed variant alongside DIAGNOSTIC_BLUE
PRIMARY_ACTION_BLUE = "#107050"  # deep surgical emerald — primary CTA fill ("Launch Workstation")
BORDER_GREY = "#E2E8F0"          # 1px card/divider borders
BORDER_STRONG = "#CBD5E1"        # button borders / emphasized divider / hover border
TEXT_MUTED = "#4B5563"           # muted description grey — secondary text/labels
TEXT_FAINT = "#6B7280"           # neutral slate — tertiary micro-copy, "normal finding" tint
MONO_TEXT = "#1E293B"            # tabular monospace readouts (DICOM tags, coordinates, latencies)
AMBER = "#D97706"                # muted amber — regulatory disclosure banner, moderate-risk badge

# Medical-brand palette (ORTHOC-style institutional hospital site) —
# used directly by the marketing landing page (hero, departments, about,
# doctors, footer); the tokens above double as the same colors under
# their clinical-workstation names so the diagnostic console and the
# marketing site read as one coherent brand.
BRAND_PRIMARY = "#107050"        # deep surgical emerald green
BRAND_PRIMARY_DARK = "#0D5C43"
BRAND_MINT = "#56B8A0"           # soft seafoam mint
BRAND_MINT_DARK = "#48A992"
SECTION_MINT_BG = "#F2FAF7"      # mint-tinted alternating section background

# Kept for backward-compat with call sites still referencing older token
# names from earlier theme passes; both now resolve into the light
# clinical palette above.
CLINICAL_SLATE = APP_BACKGROUND
SURGICAL_TEAL = SAGE_GREEN

RADIUS_SHARP = "10px"            # smooth rounded corners (8-12px) across cards/buttons
CARD_SHADOW = "0 4px 16px rgba(16, 112, 80, 0.07)"  # soft elevation shadow

RISK_LOW = "#16A34A"
RISK_MODERATE = "#D97706"
RISK_HIGH = "#DC2626"

# Recognized medical/clinical symbol (Unicode "Staff of Aesculapius"),
# not a decorative emoji — used as the sole browser-tab glyph across all
# three modules so the tab icon stays consistent with the "no playful
# emoji" mandate.
CLINICAL_GLYPH = "⚕"

MODEL_VERSION = "1.0.4"
SYSTEM_STATUS_LABEL = f"STATUS: ONLINE (LOCAL CPU)"

INSTITUTION_NAME = "Q-KNEE DIAGNOSTIC WORKSTATION"
INSTITUTION_DIVISION = f"Musculoskeletal Radiology Research Suite • Version {MODEL_VERSION}"

# Real-time clinical telemetry strip — rendered as monospace pills in the
# institutional header (see `render_telemetry_pills`).
TELEMETRY_ITEMS: Tuple[str, ...] = (
    "PACS FEED: SIMULATED",
    "ENGINE: NISQ-VQC (4-QUBIT)",
    "LATENCY: 16.3ms",
    SYSTEM_STATUS_LABEL,
)

DISCLOSURE_BANNER_TEXT = (
    "INVESTIGATIONAL DEVICE ONLY • NOT CLEARED FOR PRIMARY DIAGNOSTIC DETERMINATION • "
    "FOR ADJUNCTIVE TRIAGE ONLY • CONFIRMATORY OVER-READ REQUIRED BY BOARD-CERTIFIED RADIOLOGIST."
)

NOT_A_DEVICE_FOOTNOTE = (
    "Research prototype — not a certified medical device. Not for clinical use."
)


# --------------------------------------------------------------------------- #
# Global CSS injection
# --------------------------------------------------------------------------- #

def inject_clinical_theme() -> None:
    """Injects the shared clinical design system once per script rerun —
    sterile light palette, Inter/JetBrains-Mono typography, and the formal
    section-card/badge/masthead primitives every `render_*` function in
    `qknee.ui` builds on. Idempotent (plain CSS, safe to call on every
    rerun)."""
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
            letter-spacing: -0.01em !important;
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
            font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace !important;
            font-variant-numeric: tabular-nums;
            color: {MONO_TEXT} !important;
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
            background: {DIAGNOSTIC_BLUE};
            color: {CARD_SURFACE};
            font-weight: 800;
            font-size: 0.78rem;
            margin-right: 0.4rem;
        }}

        /* System-status pill: rigid border, static (no glow/pulse) dot */
        .qknee-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.66rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
            color: {SAGE_GREEN};
            background: {SAGE_TINT};
            border: 1px solid {SAGE_GREEN}55;
            border-radius: {RADIUS_SHARP};
            padding: 0.28rem 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }}
        .qknee-status-dot {{
            width: 0.4rem;
            height: 0.4rem;
            border-radius: 50%;
            background: {SAGE_GREEN};
        }}

        /* Real-time clinical telemetry strip — monospace pills in the header */
        .qknee-telemetry-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            justify-content: flex-end;
        }}
        .qknee-telemetry-pill {{
            display: inline-block;
            font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
            font-size: 0.64rem;
            font-weight: 600;
            color: {MONO_TEXT};
            background: {APP_BACKGROUND};
            border: 1px solid {BORDER_GREY};
            border-radius: {RADIUS_SHARP};
            padding: 0.22rem 0.5rem;
            white-space: nowrap;
            letter-spacing: 0.01em;
        }}

        /* Regulatory disclosure — rigid amber alert box, no collapsible
        expander, no accent-bar/gradient treatment */
        .qknee-disclosure-banner {{
            display: flex;
            align-items: flex-start;
            gap: 0.55rem;
            font-size: 0.74rem;
            font-weight: 700;
            line-height: 1.5;
            letter-spacing: 0.01em;
            color: #92400E;
            background: #FEF3C7;
            border: 1px solid #FCD34D;
            border-radius: {RADIUS_SHARP};
            padding: 0.65rem 0.95rem;
            margin: 0.9rem 0 1.3rem 0;
        }}
        .qknee-disclosure-banner .qknee-disclosure-glyph {{
            color: #92400E;
            font-weight: 700;
            flex-shrink: 0;
        }}
        .qknee-disclosure-banner b {{ color: #92400E; }}

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

        /* White clinical specification cards (landing page technical grid) —
        crisp white surface with soft elevation, no blur/glass/gradient */
        .qknee-card-glass {{
            background: {CARD_SURFACE};
            border: 1px solid {BORDER_GREY};
            border-radius: {RADIUS_SHARP};
            box-shadow: {CARD_SHADOW};
            padding: 20px;
            height: 100%;
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }}
        .qknee-card-glass:hover {{
            box-shadow: 0 8px 24px rgba(16, 112, 80, 0.12);
            transform: translateY(-2px);
        }}
        .qknee-card-pill {{
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: {DIAGNOSTIC_BLUE};
            background: {DIAGNOSTIC_BLUE}14;
            border: 1px solid {DIAGNOSTIC_BLUE}44;
            border-radius: {RADIUS_SHARP};
            padding: 0.2rem 0.6rem;
            margin-bottom: 0.7rem;
        }}
        .qknee-card-metric {{
            display: inline-block;
            font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
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

        /* Risk / status badges (formal, no emoji, square corners) */
        .qknee-badge {{
            display: inline-block;
            font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            border-radius: {RADIUS_SHARP};
            padding: 0.24rem 0.6rem;
        }}
        .qknee-badge-low {{ color: {RISK_LOW}; background: {RISK_LOW}14; border: 1px solid {RISK_LOW}55; }}
        .qknee-badge-moderate {{ color: {RISK_MODERATE}; background: {RISK_MODERATE}14; border: 1px solid {RISK_MODERATE}55; }}
        .qknee-badge-high {{ color: {RISK_HIGH}; background: {RISK_HIGH}14; border: 1px solid {RISK_HIGH}55; }}
        .qknee-badge-info {{ color: {DIAGNOSTIC_BLUE}; background: {DIAGNOSTIC_BLUE}14; border: 1px solid {DIAGNOSTIC_BLUE}55; }}
        .qknee-badge-neutral {{ color: {TEXT_MUTED}; background: {TEXT_FAINT}14; border: 1px solid {BORDER_GREY}; }}

        .qknee-ci {{
            font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            margin-top: 0.15rem;
        }}

        /* Buttons: clean rounded white buttons with a slate border; the
        primary CTA is a solid emerald pill with a soft mint-tinted shadow,
        matching the medical-brand hero/CTA treatment. */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid {BORDER_STRONG};
            background-color: {CARD_SURFACE};
            color: {STERILE_WHITE};
            white-space: nowrap;
            transition: background-color 0.15s ease, box-shadow 0.15s ease;
            box-shadow: none;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: {BRAND_MINT};
            color: {BRAND_PRIMARY_DARK};
            background-color: {SECTION_MINT_BG};
        }}
        .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
            background-color: {PRIMARY_ACTION_BLUE};
            border-color: {PRIMARY_ACTION_BLUE};
            color: {CARD_SURFACE};
            box-shadow: 0 4px 14px rgba(16, 112, 80, 0.22);
        }}
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {{
            background-color: {BRAND_PRIMARY_DARK};
            border-color: {BRAND_PRIMARY_DARK};
            color: {CARD_SURFACE};
            box-shadow: 0 6px 18px rgba(16, 112, 80, 0.3);
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

        /* Institutional segmented navigation control
        (qknee.ui.auth_view.render_global_navbar) — scoped to
        `st.container(key="qknee_navbar")`'s real generated wrapper class
        (Streamlit actually nests everything rendered inside a keyed
        container, unlike separate st.markdown calls, so this selector
        genuinely reaches only the nav pills). Active/inactive is
        Streamlit's own `type="primary"`/`type="secondary"`. `white-space:
        nowrap` plus a `min-width` sized to the longest label — not
        `nowrap` alone — is what actually stops "Cohort Analytics & ROC"
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
            background-color: {APP_BACKGROUND};
            border-color: {BORDER_GREY};
            color: {STERILE_WHITE};
        }}
        .st-key-qknee_navbar .stButton > button[kind="primary"] {{
            background-color: {CARD_SURFACE};
            border: 1px solid {BORDER_GREY};
            border-bottom: 2px solid {DIAGNOSTIC_BLUE};
            color: {STERILE_WHITE};
            border-radius: {RADIUS_SHARP} {RADIUS_SHARP} 0 0;
        }}
        .st-key-qknee_navbar .stButton > button[kind="primary"]:hover {{
            background-color: #F1F5F9;
            border-color: {BORDER_GREY};
            border-bottom: 2px solid {DIAGNOSTIC_BLUE};
            color: {STERILE_WHITE};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Institutional masthead + telemetry + disclosure banner
# --------------------------------------------------------------------------- #

def render_telemetry_pills(items: Tuple[str, ...] = TELEMETRY_ITEMS) -> str:
    """Builds the right-aligned row of monospace clinical-telemetry pills
    (`PACS FEED: SIMULATED`, `ENGINE: NISQ-VQC (4-QUBIT)`, etc.) as an HTML
    fragment — embedded directly into the masthead/navbar markup rather
    than rendered standalone, since Streamlit isolates each `st.markdown`
    call into its own DOM fragment."""
    pills = "".join(f'<span class="qknee-telemetry-pill">{item}</span>' for item in items)
    return f'<div class="qknee-telemetry-row">{pills}</div>'


def render_institutional_masthead(active_module: str) -> None:
    """Renders the institutional header shared by every page: formal
    laboratory branding on the left, and the real-time clinical telemetry
    strip (PACS feed / engine / latency / status) on the right."""
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
            {render_telemetry_pills()}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclosure_banner() -> None:
    """Mandatory regulatory (CE/FDA triage) disclosure — a rigid 1px-border
    amber alert box, not a collapsible expander. Called once, directly
    beneath the institutional header, by `qknee.ui.landing_page.render_hero()`
    and `qknee.ui.analysis_app.render_header()`."""
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
    cool_rgb = np.array([2, 132, 199], dtype=np.float32)     # diagnostic blue
    mid_rgb = np.array([248, 250, 252], dtype=np.float32)    # sterile white
    warm_rgb = np.array([220, 38, 38], dtype=np.float32)     # diagnostic crimson

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


_PLANE_ORIENTATION_MARKERS = {
    # (top, bottom, left, right) anatomical-orientation letters conventionally
    # shown at a PACS viewport's edges — Anterior/Posterior/Superior/Inferior.
    "axial": ("A", "P", "R", "L"),
    "coronal": ("S", "I", "R", "L"),
    "sagittal": ("S", "I", "A", "P"),
}


def draw_orientation_markers(image: np.ndarray, plane: str) -> np.ndarray:
    """Draws subtle PACS-style anatomical orientation letters (A/P/S/I) at
    the four edges of `image`, matching the convention radiologists expect
    on a diagnostic viewport. Returns a `(H, W, 3)` BGR uint8 array; does
    not modify `image` in place."""
    if image.ndim == 2:
        canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()
        if canvas.shape[2] == 1:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    top, bottom, left, right = _PLANE_ORIENTATION_MARKERS.get(plane.lower(), ("A", "P", "S", "I"))
    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, min(height, width) / 500)
    thickness = max(1, min(height, width) // 300)
    color = _CROSSHAIR_BGR
    margin = max(8, min(height, width) // 30)

    def _put(label: str, x: int, y: int) -> None:
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.putText(canvas, label, (x - tw // 2, y + th // 2), font, scale, color, thickness, cv2.LINE_AA)

    _put(top, width // 2, margin)
    _put(bottom, width // 2, height - margin)
    _put(left, margin, height // 2)
    _put(right, width - margin, height // 2)
    return canvas
