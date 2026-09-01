"""
Q-Knee shared clinical design system.

Centralizes the visual identity, copy constants, and small rendering
helpers that `qknee.ui.landing_page`, `qknee.ui.analysis_app`, and
`qknee.ui.dashboard` all build on, so the public landing page, the
PACS-style diagnostic workstation, and the clinical dashboard read as one
coherent product rather than three independently themed pages.

Palette: deep clinical slate (`#0F172A`), sterile off-white (`#F8FAFC`),
surgical teal (`#0D9488`), diagnostic blue (`#2563EB`), and cool-grey
borders (`#CBD5E1`). Typography is Inter (falling back to SF Pro
Display / system sans), no decorative emoji or cartoonish iconography —
status and risk indicators are conveyed through formal badges instead.

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import cv2
import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #

CLINICAL_SLATE = "#0F172A"
STERILE_WHITE = "#F8FAFC"
SURGICAL_TEAL = "#0D9488"
DIAGNOSTIC_BLUE = "#2563EB"
BORDER_GREY = "#CBD5E1"
TEXT_MUTED = "#94A3B8"

RISK_LOW = "#15803D"
RISK_MODERATE = "#B45309"
RISK_HIGH = "#B91C1C"

# Recognized medical/clinical symbol (Unicode "Staff of Aesculapius"),
# not a decorative emoji — used as the sole browser-tab glyph across all
# three modules so the tab icon stays consistent with the "no playful
# emoji" mandate.
CLINICAL_GLYPH = "⚕"

MODEL_VERSION = "v1.0.4-NISQ"
SYSTEM_STATUS_LABEL = f"System Online • Model {MODEL_VERSION}"

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
    palette, Inter/SF-Pro-Display typography, and the formal section-card/
    badge/masthead primitives every `render_*` function in `qknee.ui`
    builds on. Idempotent (plain CSS, safe to call on every rerun)."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp, .stMarkdown, .stButton, .stMetric {{
            font-family: 'Inter', -apple-system, 'SF Pro Display', 'Segoe UI', sans-serif !important;
        }}
        .stApp {{
            background-color: {CLINICAL_SLATE};
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 700 !important;
            line-height: 1.2 !important;
            letter-spacing: -0.02em !important;
            color: {STERILE_WHITE} !important;
        }}
        p, span, label, div {{
            line-height: 1.45;
        }}

        /* Institutional masthead */
        .qknee-masthead {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.6rem;
            padding: 0.95rem 1.3rem;
            background: #111C33;
            border: 1px solid {BORDER_GREY}30;
            border-radius: 0.5rem;
            margin-bottom: 0.7rem;
        }}
        .qknee-masthead-brand {{
            font-size: 1.05rem;
            font-weight: 800;
            color: {STERILE_WHITE};
            letter-spacing: -0.01em;
        }}
        .qknee-masthead-division {{
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-top: 0.1rem;
        }}
        .qknee-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.72rem;
            font-weight: 700;
            color: {SURGICAL_TEAL};
            background: {SURGICAL_TEAL}1A;
            border: 1px solid {SURGICAL_TEAL}55;
            border-radius: 999px;
            padding: 0.32rem 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            white-space: nowrap;
        }}
        .qknee-status-dot {{
            width: 0.4rem;
            height: 0.4rem;
            border-radius: 50%;
            background: {SURGICAL_TEAL};
            box-shadow: 0 0 0 3px {SURGICAL_TEAL}22;
        }}

        /* Disclosure banner */
        .qknee-disclosure {{
            font-size: 0.76rem;
            color: {TEXT_MUTED};
            background: #0B1220;
            border: 1px solid {BORDER_GREY}28;
            border-left: 3px solid {DIAGNOSTIC_BLUE};
            border-radius: 0.35rem;
            padding: 0.6rem 0.95rem;
            margin-bottom: 1.1rem;
        }}
        .qknee-disclosure b {{ color: {STERILE_WHITE}; }}

        /* Section cards */
        .qknee-card {{
            background: #111C33;
            border: 1px solid {BORDER_GREY}2E;
            border-radius: 0.55rem;
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
        .qknee-eyebrow {{
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: {DIAGNOSTIC_BLUE};
            margin-bottom: 0.35rem;
        }}

        /* Risk / status badges (replace emoji indicators) */
        .qknee-badge {{
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            border-radius: 0.3rem;
            padding: 0.24rem 0.6rem;
        }}
        .qknee-badge-low {{ color: {RISK_LOW}; background: {RISK_LOW}1F; border: 1px solid {RISK_LOW}55; }}
        .qknee-badge-moderate {{ color: {RISK_MODERATE}; background: {RISK_MODERATE}1F; border: 1px solid {RISK_MODERATE}55; }}
        .qknee-badge-high {{ color: {RISK_HIGH}; background: {RISK_HIGH}1F; border: 1px solid {RISK_HIGH}55; }}
        .qknee-badge-info {{ color: {DIAGNOSTIC_BLUE}; background: {DIAGNOSTIC_BLUE}1A; border: 1px solid {DIAGNOSTIC_BLUE}55; }}
        .qknee-badge-neutral {{ color: {TEXT_MUTED}; background: {TEXT_MUTED}1A; border: 1px solid {TEXT_MUTED}44; }}

        .qknee-ci {{
            font-size: 0.72rem;
            color: {TEXT_MUTED};
            margin-top: 0.15rem;
        }}

        /* Buttons: squared-off, formal */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 0.35rem;
            font-weight: 600;
            font-size: 0.85rem;
            border: 1px solid {BORDER_GREY}44;
        }}

        hr {{ border-color: {BORDER_GREY}22; }}
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
            <div>
                <div class="qknee-masthead-brand">{INSTITUTION_NAME}</div>
                <div class="qknee-masthead-division">{INSTITUTION_DIVISION} &middot; {active_module}</div>
            </div>
            <div class="qknee-status-pill"><span class="qknee-status-dot"></span>{SYSTEM_STATUS_LABEL}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclosure_banner() -> None:
    st.markdown(f'<div class="qknee-disclosure">{DISCLOSURE_BANNER_TEXT}</div>', unsafe_allow_html=True)


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
