"""
Q-Knee public landing page (Streamlit) — the marketing/onboarding entry
view wired into `qknee.ui.dashboard`'s `main()`.

"ORTHOC" hospital aesthetic: deep medical teal / slate-navy / pure-white
clinical palette, sharp typography, and pure CSS/SVG diagrams in place of
photography — every visual on this page is either a real artifact this
pipeline actually produced (a generated circuit diagram, a measured
benchmark chart, a case's own Grad-CAM overlay) or a hand-drawn CSS/SVG
schematic of the real 5-stage architecture, never stock/placeholder
imagery of an unrelated scan or clinician.

Reuses `qknee/artifacts/precomputed_cache.json` (the same Judge Fast-Path
cache `qknee.ui.dashboard`/`qknee.ui.analysis_app` serve from) for the
Demo Sample Quick-Loaders: decoding one case's pre-blended, base64-embedded
Grad-CAM overlay costs a base64 decode + PNG decode, nothing more — no
ResNet18 forward pass, no PennyLane QNode execution.

Author: Yashika Nayak

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import streamlit as st

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

_config = load_config()
logger = get_logger(__name__)

# Resolved relative to the repo root (this file's grandparent directory:
# qknee/ui/landing_page.py -> qknee/ -> repo root), never the process's
# current working directory — same convention as qknee.ui.dashboard /
# qknee.ui.analysis_app, since `streamlit run` doesn't guarantee cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS_DIR = _REPO_ROOT / "qknee" / "artifacts"
DECK_FIGURES_DIR = _ARTIFACTS_DIR / "deck_figures"
PRECOMPUTED_CACHE_PATH = _ARTIFACTS_DIR / "precomputed_cache.json"
BENCHMARK_RESULTS_PATH = _ARTIFACTS_DIR / "benchmark_results.json"

CIRCUIT_DIAGRAM_PATH = DECK_FIGURES_DIR / "circuit_diagram.png"
ROC_EFFICIENCY_PATH = DECK_FIGURES_DIR / "roc_and_parameter_efficiency.png"

TAGLINE = "Accelerating orthopedic MRI triage by coupling deep spatial feature extraction with a 4-qubit variational quantum classifier."
AUTHOR_CREDIT = "Yashika Nayak"

# Session-state keys shared with `qknee.ui.dashboard` — its `main()` reads
# `VIEW_STATE_KEY` to decide whether to render this landing page or the
# full diagnostic/benchmark console, and this module's CTA buttons write
# to it (then trigger a rerun) to navigate there.
VIEW_STATE_KEY = "qknee_active_view"
VIEW_LANDING = "landing"
VIEW_DIAGNOSTIC = "diagnostic"
VIEW_BENCHMARK = "benchmark"

# Written by the Section-4 sample presets below; read (and cleared) by
# `qknee.ui.dashboard.render_fast_path_sidebar` so a preset click carries
# straight through into the workstation with that exact case pre-loaded —
# "automatically load pre-cached volume tensors into session state for
# immediate inspection" rather than landing on the workstation empty and
# making the visitor pick the case again from a dropdown.
PRESELECTED_CASE_KEY = "qknee_preselected_fastpath_case_id"

# Demo Sample Quick-Loaders (Section 4) — mapped onto real
# `precomputed_cache.json` cases by category. The cache (see
# `scripts/generate_demo_cache.py`) only models three ground-truth
# categories (Normal / ACL Tear / Meniscal Tear), so "Complex
# Multi-compartment Defect" reuses the closest available multi-finding
# proxy (a second, distinct meniscal-tear case) rather than a fabricated
# category — the risk score/tier shown once loaded is still that case's
# own real, cached value, never invented for this preset.
SAMPLE_PRESETS: List[Dict[str, str]] = [
    {"case_id": "case_0005", "label": "Sample 01", "title": "Confirmed ACL Tear",
     "detail": "Sagittal series · full-thickness ACL discontinuity"},
    {"case_id": "case_0001", "label": "Sample 02", "title": "Normal Intact Meniscus",
     "detail": "Sagittal series · no structural abnormality"},
    {"case_id": "case_0009", "label": "Sample 03", "title": "Complex Multi-compartment Defect",
     "detail": "Sagittal series · meniscal tear with adjacent compartment involvement"},
]

# Reference/target validation figures (Section 3) — the PRD's stated
# hackathon-evaluation benchmark, shown as the headline comparison card.
# These are explicitly labeled reference/target numbers, not a live
# re-computation: `_load_benchmark_results()` below additionally surfaces
# whatever `scripts/run_benchmark.py` has actually measured on this
# machine's current (small mock) validation split, in its own clearly
# separate "Live Benchmark Run" disclosure, so a visitor is never shown a
# fabricated number presented as a live one.
REFERENCE_BENCHMARK_ROWS: List[Dict[str, str]] = [
    {"model": "Classical ResNet18 + Linear", "acl_auc": "0.884", "meniscal_auc": "0.831", "kind": "classical"},
    {"model": "Hybrid ResNet18 + 4-Qubit VQC", "acl_auc": "0.912", "meniscal_auc": "0.857", "kind": "hybrid"},
]
REFERENCE_PARAMETER_EFFICIENCY_TEXT = (
    "Quantum Hilbert-space boundary evaluation with 78% fewer dense classification weights "
    "than an equivalent classical bottleneck head."
)


# --------------------------------------------------------------------------- #
# Data loading — dynamic metric callouts + sample-preset roster
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def _load_precomputed_cases() -> List[Dict]:
    """Loads `precomputed_cache.json`'s cases list; `[]` (logged) if the
    cache hasn't been built yet (`python scripts/generate_demo_cache.py`)
    or fails to parse, so Section 4 can degrade to an explanatory message
    instead of erroring.

    `max_entries=10, ttl=3600`: this function takes no arguments (only
    ever one real entry), but capped consistently with every other raw-
    data-loading cache in this project so a stale read after the on-disk
    file changes doesn't linger indefinitely — see the 1GB-Streamlit-
    Cloud-ceiling budget this module is audited against."""
    if not PRECOMPUTED_CACHE_PATH.exists():
        logger.info("No precomputed cache found at %s; Demo Sample Quick-Loaders disabled.", PRECOMPUTED_CACHE_PATH)
        return []
    try:
        payload = json.loads(PRECOMPUTED_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read precomputed cache at %s: %s", PRECOMPUTED_CACHE_PATH, exc)
        return []
    return payload.get("cases", [])


@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def _load_benchmark_results() -> Optional[Dict]:
    """Loads `scripts/run_benchmark.py`'s exported results, if any have
    been generated yet — powers the "Live Benchmark Run" disclosure with a
    real, currently-measured figure instead of a second fabricated one.
    Same `max_entries=10, ttl=3600` capping rationale as
    `_load_precomputed_cases` above."""
    if not BENCHMARK_RESULTS_PATH.exists():
        return None
    try:
        return json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read benchmark results at %s: %s", BENCHMARK_RESULTS_PATH, exc)
        return None


def _parameter_reduction_pct() -> float:
    """Trainable-parameter compression of the 4-qubit VQC head vs. an
    equivalent classical `Linear(feature_dim, n_qubits) -> Linear(n_qubits, 2)`
    bottleneck baseline — computed from the live config, not a hardcoded
    marketing number, so this stays accurate if `config.yaml`'s
    qubit/layer count ever changes. Distinct from
    `REFERENCE_PARAMETER_EFFICIENCY_TEXT`'s PRD-quoted 78% figure (a
    different, stated-in-the-brief baseline comparison) — both are shown,
    each labeled for what it actually is."""
    feature_dim = _config.resnet.feature_dim
    n_qubits = _config.quantum.n_qubits
    linear_params = (feature_dim * n_qubits + n_qubits) + (n_qubits * 2 + 2)
    vqc_params = _config.quantum.n_layers * n_qubits * 3 + n_qubits + 1  # RX/RY/RZ per qubit per layer + readout
    return (1 - vqc_params / linear_params) * 100


def _read_b64_image(path: Path) -> Optional[str]:
    """Base64-encodes a local pipeline-artifact PNG for inline
    `<img src="data:...">` embedding inside this page's hand-built HTML
    blocks — Streamlit can't serve an arbitrary filesystem path as an HTTP
    URL, so a raw `<img src="local/path.png">` would silently render
    broken there. Returns `None` (logged) if the artifact hasn't been
    generated yet (`python scripts/generate_deck_assets.py`) so callers can
    degrade to no image rather than a broken one."""
    if not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("Failed to read image artifact %s: %s", path, exc)
        return None


def _case_card_image_src(case: Dict) -> Optional[str]:
    """A sample preset's preview thumbnail is that exact case's own
    pre-blended Grad-CAM overlay (already embedded as base64 in
    `precomputed_cache.json` by `scripts/generate_demo_cache.py`) — a real
    clinical artifact produced by this pipeline, never stand-in stock
    photography of an unrelated knee/joint. Returns `None` (caller renders
    a neutral placeholder block) if this case has no embedded heatmap."""
    heatmap_b64 = case.get("heatmap_base64")
    return f"data:image/png;base64,{heatmap_b64}" if heatmap_b64 else None


def _case_risk_pct(case: Dict) -> str:
    try:
        return f"{float(case.get('risk_score', 0.0)) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


# --------------------------------------------------------------------------- #
# Design tokens — hospital "ORTHOC" clinical palette
# --------------------------------------------------------------------------- #

TEAL = "#0D9488"          # primary — badges, active accents, primary CTA
TEAL_DARK = "#0F766E"     # primary hover / pressed state
NAVY = "#0F172A"          # neutral dark — headings, high-contrast text
WHITE = "#FFFFFF"         # neutral light — page/card surfaces
BORDER = "#E2E8F0"        # hairline card/section borders
CARD_BG = "#F8FAFC"       # subtle off-white card fill (distinct from pure-white page bg)
BODY = "#475569"          # body copy
MUTED = "#94A3B8"         # muted labels/captions
AMBER = "#D97706"         # clinical amber — alerts / moderate risk
EMERALD = "#059669"       # emerald — healthy tissue / low risk / hybrid-model accent
CRIMSON = "#DC2626"       # crimson — tear detection / high risk

_LANDING_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, .stButton, .stMetric {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}}
.stApp {{ background-color: {WHITE}; }}
[data-testid="stAppViewContainer"] .main .block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1180px !important;
}}
h1, h2, h3, h4, h5, h6 {{ color: {NAVY}; }}
p, span, label, li {{ color: {BODY}; }}

/* ---- Hero ---- */
.qknee-hero-badge {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: {TEAL}14; border: 1px solid {TEAL}44; color: {TEAL_DARK};
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
    border-radius: 999px; padding: 0.35rem 0.85rem; margin-bottom: 1.1rem;
}}
.qknee-hero-title {{
    color: {NAVY}; font-size: 2.7rem; font-weight: 800; letter-spacing: -0.02em;
    line-height: 1.08; margin: 0 0 1rem 0;
}}
.qknee-hero-subtitle {{
    color: {BODY}; font-size: 1.02rem; line-height: 1.65; max-width: 42rem; margin-bottom: 1.6rem;
}}
.st-key-qknee_hero_cta .stButton > button[kind="primary"] {{
    background: {TEAL} !important; border-color: {TEAL} !important; color: {WHITE} !important;
    font-weight: 700 !important; border-radius: 6px !important; padding: 0.7rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(13, 148, 136, 0.22) !important;
}}
.st-key-qknee_hero_cta .stButton > button[kind="primary"]:hover {{ background: {TEAL_DARK} !important; }}
.st-key-qknee_hero_cta .stButton > button[kind="secondary"] {{
    background: {WHITE} !important; border: 1px solid {BORDER} !important; color: {NAVY} !important;
    font-weight: 700 !important; border-radius: 6px !important; padding: 0.7rem 1.4rem !important;
    box-shadow: none !important;
}}
.st-key-qknee_hero_cta .stButton > button[kind="secondary"]:hover {{ border-color: {TEAL} !important; color: {TEAL_DARK} !important; }}

/* ---- Section heading ---- */
.qknee-section-eyebrow {{
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    color: {TEAL_DARK}; margin-bottom: 0.35rem;
}}
.qknee-section-title {{ color: {NAVY}; font-weight: 800; font-size: 1.55rem; margin-bottom: 0.4rem; }}
.qknee-section-subtitle {{ color: {MUTED}; font-size: 0.85rem; max-width: 42rem; margin-bottom: 1.6rem; }}

/* ---- Pipeline visualizer ---- */
.qknee-pipeline-row {{ display: flex; align-items: stretch; gap: 0.4rem; flex-wrap: wrap; }}
.qknee-pipeline-step {{
    flex: 1 1 170px; min-width: 150px;
    background: {WHITE}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 1.1rem 0.9rem; text-align: center;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}
.qknee-pipeline-icon {{
    width: 46px; height: 46px; border-radius: 50%; border: 2px solid {TEAL};
    display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem auto;
    color: {TEAL_DARK}; background: {TEAL}0D;
}}
.qknee-pipeline-step-num {{
    font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.62rem; font-weight: 700;
    color: {MUTED}; letter-spacing: 0.06em; margin-bottom: 0.3rem;
}}
.qknee-pipeline-step-title {{ font-size: 0.8rem; font-weight: 800; color: {NAVY}; margin-bottom: 0.35rem; line-height: 1.25; }}
.qknee-pipeline-step-body {{ font-size: 0.72rem; color: {BODY}; line-height: 1.5; }}
.qknee-pipeline-arrow {{
    display: flex; align-items: center; justify-content: center; flex: 0 0 28px;
    color: {MUTED};
}}
@media (max-width: 900px) {{ .qknee-pipeline-arrow {{ display: none; }} }}

/* ---- Benchmarks table ---- */
.qknee-bench-table {{
    width: 100%; border-collapse: collapse; background: {WHITE};
    border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden;
}}
.qknee-bench-table th {{
    background: {CARD_BG}; color: {MUTED}; font-size: 0.66rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em; text-align: left;
    padding: 0.65rem 0.9rem; border-bottom: 1px solid {BORDER};
}}
.qknee-bench-table td {{
    padding: 0.75rem 0.9rem; font-size: 0.85rem; color: {NAVY}; border-bottom: 1px solid {BORDER};
}}
.qknee-bench-table tr:last-child td {{ border-bottom: none; }}
.qknee-bench-table tr.hybrid-row {{ background: {EMERALD}0A; }}
.qknee-bench-auc {{
    font-family: 'JetBrains Mono', Consolas, monospace; font-weight: 700;
}}
.qknee-bench-auc.better {{ color: {EMERALD}; }}
.qknee-bench-footnote {{
    font-size: 0.72rem; color: {MUTED}; margin-top: 0.6rem; line-height: 1.6;
}}
.qknee-efficiency-card {{
    background: {NAVY}; border-radius: 10px; padding: 1.1rem 1.3rem; margin-top: 1rem;
    display: flex; align-items: center; gap: 0.8rem;
}}
.qknee-efficiency-card span.pct {{
    font-family: 'JetBrains Mono', Consolas, monospace; font-size: 1.5rem; font-weight: 800; color: {TEAL};
}}
.qknee-efficiency-card span.text {{ color: #CBD5E1; font-size: 0.8rem; line-height: 1.5; }}

/* ---- Demo sample quick-loaders ---- */
.qknee-sample-card {{
    background: {WHITE}; border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; height: 100%;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}}
.qknee-sample-thumb {{ width: 100%; height: 120px; object-fit: cover; display: block; background: {CARD_BG}; }}
.qknee-sample-thumb-placeholder {{
    width: 100%; height: 120px; display: flex; align-items: center; justify-content: center;
    background: {CARD_BG}; color: {MUTED}; font-size: 0.7rem;
}}
.qknee-sample-body {{ padding: 0.9rem 1rem 0.3rem 1rem; }}
.qknee-sample-label {{
    font-family: 'JetBrains Mono', Consolas, monospace; font-size: 0.62rem; font-weight: 700;
    color: {TEAL_DARK}; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem;
}}
.qknee-sample-title {{ font-size: 0.92rem; font-weight: 800; color: {NAVY}; margin-bottom: 0.2rem; }}
.qknee-sample-detail {{ font-size: 0.72rem; color: {MUTED}; margin-bottom: 0.7rem; }}
.st-key-qknee_samples .stButton > button {{
    background: {WHITE} !important; border: 1px solid {BORDER} !important; color: {NAVY} !important;
    border-radius: 0 0 10px 10px !important; font-weight: 700 !important; box-shadow: none !important;
    border-top: none !important; margin-top: -0.6rem !important;
}}
.st-key-qknee_samples .stButton > button:hover {{ background: {TEAL}0D !important; border-color: {TEAL} !important; color: {TEAL_DARK} !important; }}
[data-testid="stMetricValue"] {{ color: {NAVY} !important; font-family: 'JetBrains Mono', Consolas, monospace !important; }}
[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}

/* ---- Footer ---- */
.qknee-footer {{ background: {NAVY}; border-radius: 10px; padding: 2rem 2rem 1.2rem 2rem; margin-top: 1.5rem; }}
.qknee-footer h4 {{ color: {WHITE}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.7rem; }}
.qknee-footer p, .qknee-footer li, .qknee-footer a {{ font-size: 0.78rem; color: rgba(255,255,255,0.72); line-height: 1.85; text-decoration: none; }}
.qknee-footer a:hover {{ color: {TEAL}; }}
.qknee-footer ul {{ list-style: none; padding: 0; margin: 0; }}
.qknee-footer-bottom {{
    border-top: 1px solid rgba(255,255,255,0.12); margin-top: 1.2rem; padding-top: 0.9rem;
    font-size: 0.7rem; font-weight: 600; color: rgba(255,255,255,0.8); text-align: center;
}}
</style>
"""


def inject_orthoc_theme() -> None:
    """Injects this page's base stylesheet once per script rerun —
    idempotent plain CSS, safe to call more than once. Exposed at module
    level (rather than only inside `render_landing_page`) so
    `streamlit_app.py` can also invoke it — see that module's docstring
    for why it does so *after* `dashboard.main()` returns rather than
    before (Streamlit requires `st.set_page_config()`, called inside
    `dashboard.render_header()`, to be the very first Streamlit command in
    a script run; a `<style>` tag's rules apply to the whole document
    regardless of where in the DOM it's inserted, so injecting it last in
    the same rerun still restyles everything above it)."""
    st.markdown(_LANDING_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small inline-SVG glyphs for the pipeline visualizer (no icon font/package,
# `currentColor` stroke so each step tints purely via the card's CSS color)
# --------------------------------------------------------------------------- #

_STEP_ICON_PATHS: Dict[str, str] = {
    "volume": (
        '<polygon points="12,3 21,7.5 21,16.5 12,21 3,16.5 3,7.5"/>'
        '<polyline points="3,7.5 12,12 21,7.5"/><line x1="12" y1="12" x2="12" y2="21"/>'
    ),
    "backbone": (
        '<rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/>'
        '<rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/>'
        '<line x1="10" y1="7" x2="14" y2="7"/><line x1="10" y1="17" x2="14" y2="17"/>'
    ),
    "bottleneck": (
        '<line x1="2" y1="6" x2="10" y2="6"/><line x1="2" y1="12" x2="10" y2="12"/>'
        '<line x1="2" y1="18" x2="10" y2="18"/><path d="M10 4 L18 12 L10 20"/><line x1="18" y1="12" x2="22" y2="12"/>'
    ),
    "circuit": (
        '<line x1="2" y1="6" x2="22" y2="6"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="18" x2="22" y2="18"/>'
        '<rect x="6" y="3" width="5" height="6" rx="1"/><rect x="13" y="9" width="5" height="6" rx="1"/>'
        '<rect x="6" y="15" width="5" height="6" rx="1"/>'
    ),
    "crosshair": (
        '<circle cx="12" cy="12" r="8"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>'
        '<line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>'
    ),
    "chevron": '<polyline points="9,4 17,12 9,20"/>',
    "search": '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
}


def _svg(name: str, size: int = 22, stroke_width: float = 1.8, color: Optional[str] = None) -> str:
    body = _STEP_ICON_PATHS.get(name, "")
    style = f'style="vertical-align:middle; color:{color};"' if color else 'style="vertical-align:middle;"'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" {style}>{body}</svg>'
    )


# --------------------------------------------------------------------------- #
# Section 1 — Institutional Header & Hero
# --------------------------------------------------------------------------- #

def render_hero() -> None:
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="qknee-hero-badge">CLINICAL EVALUATION RELEASE &bull; NISQ-READY HYBRID ML</div>',
                unsafe_allow_html=True)
    st.markdown('<h1 class="qknee-hero-title">Q-Knee Diagnostic Platform</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="qknee-hero-subtitle">{TAGLINE}</p>', unsafe_allow_html=True)

    with st.container(key="qknee_hero_cta"):
        cta_col1, cta_col2, _ = st.columns([1.7, 1.7, 2.6])
        with cta_col1:
            if st.button("Launch Diagnostic Workstation", key="qknee_hero_launch", type="primary",
                          use_container_width=True):
                st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
                st.rerun()
        with cta_col2:
            st.markdown(
                '<a href="#architecture" style="text-decoration:none;">'
                '<div style="text-align:center; border:1px solid #E2E8F0; border-radius:6px; '
                'padding:0.7rem 1.2rem; font-weight:700; color:#0F172A; font-size:0.9rem;">'
                'Inspect Architecture &amp; Benchmarks</div></a>',
                unsafe_allow_html=True,
            )
    st.write("")


# --------------------------------------------------------------------------- #
# Section 2 — Interactive Hybrid Pipeline Visualizer (pure CSS/SVG)
# --------------------------------------------------------------------------- #

def render_pipeline_visualizer() -> None:
    st.markdown('<div id="architecture"></div>', unsafe_allow_html=True)
    st.markdown('<div class="qknee-section-eyebrow">System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="qknee-section-title">Hybrid Quantum-Classical Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qknee-section-subtitle">Every MRI slice flows through five deterministic stages — '
        'no step is a black box.</div>',
        unsafe_allow_html=True,
    )

    n_qubits = _config.quantum.n_qubits
    feature_dim = _config.resnet.feature_dim

    steps = [
        ("volume", "01", "3D MRI Volume", "Stanford MRNet — Sagittal / Coronal / Axial series, normalized to 128&times;128."),
        ("backbone", "02", "ResNet-18 Backbone", f"Frozen convolutional feature extractor &rarr; {feature_dim}-D latent vector per slice."),
        ("bottleneck", "03", "Linear Bottleneck", f"Dense compression &rarr; {n_qubits} orthogonal scalars, angle-scaled to [0, 2&pi;)."),
        ("circuit", "04", f"{n_qubits}-Qubit VQC", "Angle encoding (RY rotations) + entangling CNOT layers, PennyLane <code>default.qubit</code>."),
        ("crosshair", "05", "Pauli-Z Score &amp; Grad-CAM", "Per-qubit &#10216;Z&#10217; expectation &rarr; triage score, paired with a Layer-4 Grad-CAM heatmap."),
    ]

    row_html = ['<div class="qknee-pipeline-row">']
    for i, (icon_name, num, title, body) in enumerate(steps):
        row_html.append(
            f"""
            <div class="qknee-pipeline-step">
                <div class="qknee-pipeline-step-num">STEP {num}</div>
                <div class="qknee-pipeline-icon">{_svg(icon_name, size=22)}</div>
                <div class="qknee-pipeline-step-title">{title}</div>
                <div class="qknee-pipeline-step-body">{body}</div>
            </div>
            """
        )
        if i < len(steps) - 1:
            row_html.append(f'<div class="qknee-pipeline-arrow">{_svg("chevron", size=18)}</div>')
    row_html.append("</div>")
    st.markdown("".join(row_html), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Section 3 — Verified Clinical Benchmarks
# --------------------------------------------------------------------------- #

def render_benchmarks() -> None:
    st.write("")
    st.markdown('<div class="qknee-section-eyebrow">Validation</div>', unsafe_allow_html=True)
    st.markdown('<div class="qknee-section-title">Verified Clinical Benchmarks</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qknee-section-subtitle">Reference performance targets for this architecture on the '
        'MRNet-style validation cohort — a straight, honest comparison, no cherry-picked run.</div>',
        unsafe_allow_html=True,
    )

    rows_html = []
    for row in REFERENCE_BENCHMARK_ROWS:
        row_cls = "hybrid-row" if row["kind"] == "hybrid" else ""
        auc_cls = "better" if row["kind"] == "hybrid" else ""
        rows_html.append(
            f"""
            <tr class="{row_cls}">
                <td>{row['model']}</td>
                <td class="qknee-bench-auc {auc_cls}">{row['acl_auc']}</td>
                <td class="qknee-bench-auc {auc_cls}">{row['meniscal_auc']}</td>
            </tr>
            """
        )
    st.markdown(
        f"""
        <table class="qknee-bench-table">
            <thead><tr><th>Model</th><th>ACL Tear AUC</th><th>Meniscal Tear AUC</th></tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="qknee-efficiency-card">
            <span class="pct">&minus;78%</span>
            <span class="text">{REFERENCE_PARAMETER_EFFICIENCY_TEXT}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qknee-bench-footnote">Reference figures per the project evaluation brief for this '
        'architecture class; not re-run live on every page load.</div>',
        unsafe_allow_html=True,
    )

    live_results = _load_benchmark_results()
    with st.expander("Live Benchmark Run (this machine's current validation split)"):
        if live_results is None:
            st.info(
                f"No live benchmark run found at `{BENCHMARK_RESULTS_PATH}`. Run "
                "`python scripts/run_benchmark.py` to generate one."
            )
        else:
            models = live_results.get("models", [])
            dataset_info = live_results.get("dataset", {})
            st.caption(
                f"Generated {live_results.get('generated_at', 'unknown')} &middot; dataset: "
                f"{dataset_info.get('source', '?')} ({dataset_info.get('n_test', '?')} test samples, "
                f"{dataset_info.get('plane', '?')} plane) — a small mock/dev split, not the full cohort."
            )
            for model in models:
                st.markdown(
                    f"**{model.get('name', 'model')}** — ROC-AUC `{model.get('roc_auc', 0):.3f}`, "
                    f"F1 `{model.get('f1_score', 0):.3f}`, "
                    f"latency `{model.get('latency_ms_per_sample', 0):.2f} ms/sample`"
                )
    computed_reduction = _parameter_reduction_pct()
    st.caption(
        f"This deployment's live-computed parameter reduction (VQC head vs. an equivalent classical "
        f"bottleneck, from `config.yaml`'s current qubit/layer count): **{computed_reduction:.0f}%**."
    )


# --------------------------------------------------------------------------- #
# Section 4 — Demo Sample Quick-Loaders
# --------------------------------------------------------------------------- #

def render_sample_loaders() -> None:
    st.write("")
    st.markdown('<div id="samples"></div>', unsafe_allow_html=True)
    st.markdown('<div class="qknee-section-eyebrow">Instant Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="qknee-section-title">Demo Sample Quick-Loaders</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qknee-section-subtitle">One click loads that exact pre-cached case straight into the '
        'Diagnostic Workstation — zero model load, zero QNode execution.</div>',
        unsafe_allow_html=True,
    )

    all_cases = _load_precomputed_cases()
    by_id = {case["case_id"]: case for case in all_cases}

    if not all_cases:
        st.info(
            "No precomputed cache found yet. Run `python scripts/generate_demo_cache.py` to build the "
            "sample roster."
        )
        return

    with st.container(key="qknee_samples"):
        cols = st.columns(len(SAMPLE_PRESETS))
        for col, preset in zip(cols, SAMPLE_PRESETS):
            case = by_id.get(preset["case_id"])
            with col:
                if case is None:
                    st.markdown(
                        f'<div class="qknee-sample-card"><div class="qknee-sample-thumb-placeholder">'
                        f'Case unavailable</div><div class="qknee-sample-body">'
                        f'<div class="qknee-sample-label">{preset["label"]}</div>'
                        f'<div class="qknee-sample-title">{preset["title"]}</div></div></div>',
                        unsafe_allow_html=True,
                    )
                    continue

                image_src = _case_card_image_src(case)
                thumb_html = (
                    f'<img class="qknee-sample-thumb" src="{image_src}" alt="{preset["title"]} Grad-CAM preview">'
                    if image_src else
                    '<div class="qknee-sample-thumb-placeholder">Preview unavailable</div>'
                )
                st.markdown(
                    f"""
                    <div class="qknee-sample-card">
                        {thumb_html}
                        <div class="qknee-sample-body">
                            <div class="qknee-sample-label">{preset['label']} &bull; {_case_risk_pct(case)} risk</div>
                            <div class="qknee-sample-title">{preset['title']}</div>
                            <div class="qknee-sample-detail">{preset['detail']}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("Load Case", key=f"qknee_sample_load_{preset['case_id']}", use_container_width=True):
                    st.session_state[PRESELECTED_CASE_KEY] = preset["case_id"]
                    st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
                    st.rerun()


# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #

def render_footer() -> None:
    st.markdown('<div id="contact"></div>', unsafe_allow_html=True)
    st.markdown('<div class="qknee-footer">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <h4>Q-Knee Diagnostic Platform</h4>
            <p>NISQ-Ready Hybrid Quantum ML for Knee Abnormality Triage<br>
            Musculoskeletal Radiology Research Suite<br>Author: {AUTHOR_CREDIT}</p>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <h4>Research References</h4>
            <ul>
                <li><a href="https://stanfordmlgroup.github.io/competitions/mrnet/" target="_blank" rel="noopener">Stanford MRNet Study &rarr;</a></li>
                <li><a href="https://docs.pennylane.ai/" target="_blank" rel="noopener">PennyLane Documentation &rarr;</a></li>
                <li><a href="#architecture">System Architecture</a></li>
                <li><a href="#samples">Demo Samples</a></li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <h4>Quick Navigation</h4>
            <ul>
                <li><a href="#top">Home</a></li>
                <li><a href="#architecture">Architecture &amp; Benchmarks</a></li>
                <li><a href="#samples">Sample Cases</a></li>
                <li><a href="#contact">Contact</a></li>
            </ul>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="qknee-footer-bottom">
            INVESTIGATIONAL USE ONLY &bull; Stanford MRNet Validation Cohort &bull;
            Confirmatory radiologist over-read required before any clinical release.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def render_landing_page() -> None:
    """Renders the full redesigned landing page: hero, the interactive
    hybrid-pipeline visualizer, the verified clinical benchmarks table,
    and the demo sample quick-loaders. Call this from
    `qknee.ui.dashboard.main()` when `st.session_state[VIEW_STATE_KEY]`
    is `VIEW_LANDING` (its default)."""
    inject_orthoc_theme()
    render_hero()
    st.divider()
    render_pipeline_visualizer()
    render_benchmarks()
    st.divider()
    render_sample_loaders()
    st.write("")
    render_footer()
