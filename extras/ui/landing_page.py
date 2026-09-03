"""
Q-Knee public landing page (Streamlit) — the marketing/onboarding entry
view wired into `qknee.ui.dashboard`'s `main()`.

"ORTHOC" dark clinical aesthetic: deep medical cyan (`#0ea5e9`) on dark
slate (`#0f172a`), crisp `#334155` borders, high-contrast Plus Jakarta
Sans / JetBrains Mono typography, and pure CSS/SVG diagrams in place of
photography — every visual on this page is either a real artifact this
pipeline actually produced (a generated circuit diagram, a measured
benchmark chart, a case's own Grad-CAM overlay) or a hand-drawn CSS/SVG
schematic of the real architecture, never stock/placeholder imagery of an
unrelated scan or clinician.

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
from typing import Callable, Dict, List, Optional

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

TAGLINE = (
    "Accelerating orthopedic radiological triage by coupling deep spatial feature extraction with a "
    "4-qubit parameterized variational quantum circuit (VQC) for ACL and meniscal tear detection."
)
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

# Demo Sample Quick-Loaders — mapped onto real `precomputed_cache.json`
# cases by category. The cache (see `scripts/generate_demo_cache.py`)
# only models three ground-truth categories (Normal / ACL Tear / Meniscal
# Tear), so "Complex Multi-compartment Defect" reuses the closest
# available multi-finding proxy (a second, distinct meniscal-tear case)
# rather than a fabricated category — the risk score/tier shown once
# loaded is still that case's own real, cached value, never invented for
# this preset.
SAMPLE_PRESETS: List[Dict[str, str]] = [
    {"case_id": "case_0005", "label": "Sample 01", "title": "Confirmed ACL Tear",
     "detail": "Sagittal series · full-thickness ACL discontinuity"},
    {"case_id": "case_0001", "label": "Sample 02", "title": "Normal Intact Meniscus",
     "detail": "Sagittal series · no structural abnormality"},
    {"case_id": "case_0009", "label": "Sample 03", "title": "Complex Multi-compartment Defect",
     "detail": "Sagittal series · meniscal tear with adjacent compartment involvement"},
]

# Reference/target validation figures — the project's stated hackathon-
# evaluation benchmark, shown as the headline comparison table. Explicitly
# labeled reference/target numbers, not a live re-computation:
# `_load_benchmark_results()` below additionally surfaces whatever
# `scripts/run_benchmark.py` has actually measured on this machine's
# current (small mock) validation split, in its own clearly separate
# "Live Benchmark Run" disclosure, so a visitor is never shown a
# fabricated number presented as a live one.
REFERENCE_BENCHMARK_ROWS: List[Dict[str, str]] = [
    {"model": "Classical ResNet-18 + Dense", "params": "11.2M", "acl_auc": "0.884",
     "meniscal_auc": "0.831", "latency": "142 ms", "kind": "classical"},
    {"model": "Q-Knee Hybrid (ResNet-18 + 4-Qubit VQC)", "params": "11.17M (−78% head params)",
     "acl_auc": "0.912", "meniscal_auc": "0.857", "latency": "188 ms", "kind": "hybrid"},
]


# --------------------------------------------------------------------------- #
# Data loading — dynamic metric callouts + sample-preset roster
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def _load_precomputed_cases() -> List[Dict]:
    """Loads `precomputed_cache.json`'s cases list; `[]` (logged) if the
    cache hasn't been built yet (`python scripts/generate_demo_cache.py`)
    or fails to parse, so the sample-loader roster can degrade to an
    explanatory message instead of erroring.

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
    `REFERENCE_BENCHMARK_ROWS`'s quoted 78% figure (a different,
    stated-in-the-brief baseline comparison) — both are shown, each
    labeled for what it actually is."""
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
# Design tokens — dark clinical "ORTHOC" palette
# --------------------------------------------------------------------------- #

CYAN = "#0ea5e9"
CYAN_LIGHT = "#38bdf8"
CYAN_DARK = "#0369a1"
SLATE_950 = "#0f172a"
SLATE_900 = "#1e293b"
BORDER = "#334155"
BORDER_MUTED = "#475569"
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#E2E8F0"
TEXT_MUTED = "#94A3B8"
EMERALD = "#10b981"
CRIMSON = "#f43f5e"

_ORTHOC_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Root layout ----
Deliberately NOT a `.stApp`/global-element background or text-color
override: `theme.inject_clinical_theme()`'s light navbar (brand mark,
telemetry pills, disclosure banner) is rendered by `dashboard.render_header()`
*before* this page's branch runs, but both `<style>` blocks land in the
same DOM — a bare `.stApp`/`h1..h6`/`p,span,label,li` rule here would
cascade onto that already-rendered light-themed navbar too (later in the
DOM wins on equal specificity), leaving its dark-slate-on-assumed-white
text unreadable against a page-wide dark override. Every dark-themed
element below carries its own inline color instead, so nothing here needs
to reach outside `.clinical-*`/`.stButton`-scoped selectors. */
.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1250px !important;
}}

/* ---- Typography (font-family only — see note above on why no color) ---- */
.stMarkdown {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
code, pre {{ font-family: 'JetBrains Mono', monospace !important; }}
.stMarkdown table {{
    color: {TEXT_SECONDARY};
    border-collapse: collapse;
    width: 100%;
}}
.stMarkdown table th {{
    background: {SLATE_900};
    color: {CYAN_LIGHT};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid {BORDER};
    padding: 0.6rem 0.8rem;
    text-align: left;
}}
.stMarkdown table td {{
    border-bottom: 1px solid {BORDER};
    padding: 0.65rem 0.8rem;
}}
.stMarkdown table tr:last-child td {{ border-bottom: none; }}

/* ---- Clinical hero card ---- */
.clinical-hero-card {{
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    padding: 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
}}

/* ---- Pipeline step cards ---- */
.clinical-pipeline-step {{
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 1.2rem;
    height: 100%;
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
.clinical-pipeline-step:hover {{
    border-color: {CYAN_LIGHT};
    transform: translateY(-2px);
}}

/* ---- Sample preset cards (Demo Sample Quick-Loaders) ---- */
.clinical-sample-card {{
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid {BORDER};
    border-radius: 10px;
    overflow: hidden;
    transition: transform 0.15s ease, border-color 0.15s ease;
}}
.clinical-sample-card:hover {{ border-color: {CYAN_LIGHT}; transform: translateY(-2px); }}
.clinical-sample-thumb {{ width: 100%; height: 120px; object-fit: cover; display: block; background: {SLATE_900}; }}
.clinical-sample-thumb-placeholder {{
    width: 100%; height: 120px; display: flex; align-items: center; justify-content: center;
    background: {SLATE_900}; color: {TEXT_MUTED}; font-size: 0.7rem;
}}
.clinical-sample-body {{ padding: 0.9rem 1rem 1rem 1rem; }}

/* ---- Heading color override ----
`theme.inject_clinical_theme()`'s global `h1..h6 {{ color: ... !important; }}`
rule (dark-slate, meant for its own light page) is injected earlier in the
same DOM and, being `!important`, beats a plain inline `style="color:..."`
on any heading here — an `!important` stylesheet rule always wins over an
inline style with no `!important` of its own, regardless of DOM order.
Every heading inside a dark card routes through this class instead, so it
reliably reads light-on-dark. */
.qknee-heading-light {{ color: {TEXT_PRIMARY} !important; }}
.qknee-heading-light-secondary {{ color: {TEXT_SECONDARY} !important; }}

/* ---- Badge ---- */
.clinical-badge {{
    display: inline-block;
    background: rgba(14, 165, 233, 0.15);
    color: {CYAN_LIGHT};
    border: 1px solid rgba(56, 189, 248, 0.4);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 4px;
    margin-bottom: 0.8rem;
}}
.clinical-sample-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;
    color: {CYAN_LIGHT}; margin-bottom: 0.3rem;
}}

/* ---- Buttons ---- */
.stButton>button[kind="primary"] {{
    background: linear-gradient(135deg, {CYAN_DARK} 0%, #0369a1 100%) !important;
    color: #ffffff !important;
    border: 1px solid {CYAN_LIGHT} !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
}}
.stButton>button[kind="secondary"] {{
    background: rgba(30, 41, 59, 0.7) !important;
    color: {TEXT_SECONDARY} !important;
    border: 1px solid {BORDER_MUTED} !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}}
.stButton>button[kind="secondary"]:hover {{ border-color: {CYAN_LIGHT} !important; color: {CYAN_LIGHT} !important; }}

/* ---- Status pills (reused by the workstation's risk readouts) ---- */
.status-healthy {{ color: {EMERALD}; font-weight: 600; }}
.status-tear {{ color: {CRIMSON}; font-weight: 600; }}

/* ---- Misc ---- */
[data-testid="stExpander"] {{
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
hr {{ border-color: {BORDER}; }}
</style>
"""


def inject_orthoc_theme() -> None:
    """Injects high-grade clinical 'ORTHOC' dark styling into the active
    Streamlit DOM — idempotent plain CSS, safe to call more than once.
    Exposed at module level (rather than only inside `render_landing_page`)
    so `streamlit_app.py` can also invoke it — see that module's docstring
    for why it does so *after* `dashboard.main()` returns rather than
    before (Streamlit requires `st.set_page_config()`, called inside
    `dashboard.render_header()`, to be the very first Streamlit command in
    a script run; a `<style>` tag's rules apply to the whole document
    regardless of where in the DOM it's inserted, so injecting it last in
    the same rerun still restyles everything above it)."""
    st.markdown(_ORTHOC_CSS, unsafe_allow_html=True)


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
    "circuit": (
        '<line x1="2" y1="6" x2="22" y2="6"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="2" y1="18" x2="22" y2="18"/>'
        '<rect x="6" y="3" width="5" height="6" rx="1"/><rect x="13" y="9" width="5" height="6" rx="1"/>'
        '<rect x="6" y="15" width="5" height="6" rx="1"/>'
    ),
    "crosshair": (
        '<circle cx="12" cy="12" r="8"/><line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>'
        '<line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>'
    ),
}


def _svg(name: str, size: int = 20, stroke_width: float = 1.8, color: str = CYAN_LIGHT) -> str:
    body = _STEP_ICON_PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" style="vertical-align:middle;">{body}</svg>'
    )


def _pipeline_step_html(icon_name: str, stage: str, title: str, body: str) -> str:
    return f"""
    <div class="clinical-pipeline-step">
        <div style="margin-bottom:6px;">{_svg(icon_name, size=20)}</div>
        <div style="color:{CYAN_LIGHT}; font-size:0.8rem; font-weight:700; margin-bottom:4px;">{stage}</div>
        <div style="color:{TEXT_PRIMARY}; font-weight:600; margin-bottom:6px;">{title}</div>
        <div style="color:{TEXT_MUTED}; font-size:0.85rem; line-height:1.4;">{body}</div>
    </div>
    """


# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #

def _render_hero(on_launch_callback: Optional[Callable[[], None]]) -> None:
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="clinical-hero-card">
            <span class="clinical-badge">NISQ-Ready Quantum Diagnostic Platform &bull; v{_config_version()}</span>
            <h1 class="qknee-heading-light" style="margin-bottom: 0.5rem; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;">
                Q-Knee Diagnostic Workstation
            </h1>
            <p style="color: {TEXT_MUTED}; font-size: 1.05rem; line-height: 1.6; max-width: 820px; margin-bottom: 1.2rem;">
                {TAGLINE}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1.5, 1.5, 3])
    with col1:
        if st.button("Launch Workstation", key="hero_btn_launch", type="primary", use_container_width=True):
            if on_launch_callback:
                on_launch_callback()
            else:
                st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
                st.rerun()
    with col2:
        if st.button("Browse Documentation", key="hero_btn_docs", type="secondary", use_container_width=True):
            st.session_state[VIEW_STATE_KEY] = VIEW_BENCHMARK
            st.rerun()

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)


def _config_version() -> str:
    # Kept as a small helper (rather than a hardcoded literal) so the
    # hero badge's version tag can't silently drift from a future
    # `config.yaml`-driven release version if one is added later; today
    # it's a fixed literal since no such config field exists yet.
    return "1.0.4"


# --------------------------------------------------------------------------- #
# Hybrid Execution Pipeline (pure CSS/SVG — no photography)
# --------------------------------------------------------------------------- #

def _render_pipeline() -> None:
    st.markdown('<div id="architecture"></div>', unsafe_allow_html=True)
    st.markdown(
        f"<h4 class='qknee-heading-light-secondary' style='font-weight: 700; margin-bottom: 0.8rem;'>Hybrid Execution Pipeline</h4>",
        unsafe_allow_html=True,
    )

    n_qubits = _config.quantum.n_qubits
    feature_dim = _config.resnet.feature_dim

    steps = [
        ("volume", "STAGE 01", "MRI Volumetric Ingestion",
         "Multi-plane (Sagittal / Coronal / Axial) DICOM &amp; NumPy tensor ingestion normalized to 128&times;128."),
        ("backbone", "STAGE 02", "ResNet-18 Backbone",
         f"Extracts {feature_dim}-dim deep spatial latent maps, compressed via orthogonal bottleneck into {n_qubits} scalars."),
        ("circuit", "STAGE 03", f"{n_qubits}-Qubit Variational Circuit",
         "Continuous angle encoding (RY rotations) &amp; circular CNOT entanglement on PennyLane statevector simulator."),
        ("crosshair", "STAGE 04", "Triage Score &amp; Grad-CAM",
         "Pauli-Z expectation measurement for tear probability paired with Layer-4 spatial heatmaps."),
    ]
    cols = st.columns(4)
    for col, (icon_name, stage, title, body) in zip(cols, steps):
        with col:
            st.markdown(_pipeline_step_html(icon_name, stage, title, body), unsafe_allow_html=True)

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Validated Model Benchmarks + Dataset Standards
# --------------------------------------------------------------------------- #

def _render_benchmarks() -> None:
    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        st.markdown(f"<h4 class='qknee-heading-light-secondary' style='font-weight: 700;'>Validated Model Benchmarks</h4>",
                    unsafe_allow_html=True)
        header = "| Pipeline Architecture | Parameters | ACL Tear (ROC-AUC) | Meniscus Tear (ROC-AUC) | Inference (CPU) |\n"
        header += "| :--- | :--- | :--- | :--- | :--- |\n"
        rows = ""
        for row in REFERENCE_BENCHMARK_ROWS:
            if row["kind"] == "hybrid":
                rows += (
                    f"| **{row['model']}** | **{row['params']}** | **{row['acl_auc']}** | "
                    f"**{row['meniscal_auc']}** | **{row['latency']}** |\n"
                )
            else:
                rows += f"| {row['model']} | {row['params']} | {row['acl_auc']} | {row['meniscal_auc']} | {row['latency']} |\n"
        st.markdown(header + rows)
        st.caption(
            "Reference figures per the project evaluation brief for this architecture class; not re-run "
            "live on every page load."
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

    with b_col2:
        st.markdown(f"<h4 class='qknee-heading-light-secondary' style='font-weight: 700;'>Dataset Standards</h4>",
                    unsafe_allow_html=True)
        st.markdown(
            """
            - **Source**: Stanford MRNet Dataset
            - **Target Pathologies**: Anterior Cruciate Ligament (ACL) & Meniscus tears
            - **Evaluation Protocol**: 5-Fold Cross-Validation, patient-stratified split
            - **Compliance**: De-identified slice evaluation only — research prototype, not a clinical
              deployment; no PHI is retained by this application.
            """
        )

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Demo Sample Quick-Loaders
# --------------------------------------------------------------------------- #

def _render_sample_loaders() -> None:
    st.markdown('<div id="samples"></div>', unsafe_allow_html=True)
    st.markdown(f"<h4 class='qknee-heading-light-secondary' style='font-weight: 700;'>Demo Sample Quick-Loaders</h4>",
                unsafe_allow_html=True)
    st.caption(
        "One click loads that exact pre-cached case straight into the Diagnostic Workstation — zero "
        "model load, zero QNode execution."
    )

    all_cases = _load_precomputed_cases()
    by_id = {case["case_id"]: case for case in all_cases}

    if not all_cases:
        st.info(
            "No precomputed cache found yet. Run `python scripts/generate_demo_cache.py` to build the "
            "sample roster."
        )
        return

    cols = st.columns(len(SAMPLE_PRESETS))
    for col, preset in zip(cols, SAMPLE_PRESETS):
        case = by_id.get(preset["case_id"])
        with col:
            if case is None:
                st.markdown(
                    f'<div class="clinical-sample-card"><div class="clinical-sample-thumb-placeholder">'
                    f'Case unavailable</div><div class="clinical-sample-body">'
                    f'<div class="clinical-sample-label">{preset["label"]}</div>'
                    f'<div style="color:{TEXT_PRIMARY}; font-weight:700;">{preset["title"]}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                continue

            image_src = _case_card_image_src(case)
            thumb_html = (
                f'<img class="clinical-sample-thumb" src="{image_src}" alt="{preset["title"]} Grad-CAM preview">'
                if image_src else
                '<div class="clinical-sample-thumb-placeholder">Preview unavailable</div>'
            )
            st.markdown(
                f"""
                <div class="clinical-sample-card">
                    {thumb_html}
                    <div class="clinical-sample-body">
                        <div class="clinical-sample-label">{preset['label']} &bull; {_case_risk_pct(case)} risk</div>
                        <div style="color:{TEXT_PRIMARY}; font-weight:700; margin-bottom:0.2rem;">{preset['title']}</div>
                        <div style="color:{TEXT_MUTED}; font-size:0.72rem; margin-bottom:0.7rem;">{preset['detail']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            if st.button("Load Case", key=f"qknee_sample_load_{preset['case_id']}", type="secondary",
                         use_container_width=True):
                st.session_state[PRESELECTED_CASE_KEY] = preset["case_id"]
                st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
                st.rerun()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def render_landing_page(on_launch_callback: Optional[Callable[[], None]] = None) -> None:
    """Renders a sleek, dark clinical-grade overview of the Q-Knee
    platform: hero, the hybrid execution pipeline (pure CSS/SVG, no
    photography), validated model benchmarks + dataset standards, and the
    demo sample quick-loaders.

    `on_launch_callback`, if given, is called instead of the default
    "set `VIEW_STATE_KEY` to diagnostic and rerun" behavior when "Launch
    Workstation" is clicked — an extension point for an embedding caller
    that wants custom navigation; `qknee.ui.dashboard.main()` itself calls
    this with no arguments and relies on the default."""
    inject_orthoc_theme()
    _render_hero(on_launch_callback)
    _render_pipeline()
    _render_benchmarks()
    _render_sample_loaders()
