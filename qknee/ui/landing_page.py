"""
Q-Knee public landing page (Streamlit) — the marketing/onboarding entry
view wired into `qknee.ui.dashboard`'s `main()`. A prospective user or
hackathon judge lands here first: a hero section explains what Q-Knee is,
an interactive 3-step explainer walks through the ResNet18 -> quantum
circuit -> Grad-CAM/report pipeline, and a "Live Sample Showcase" lets
them preview 3 pre-scored knee MRI cases with one click — no login, no
upload, no cold model/QNode load.

Reuses `qknee/artifacts/precomputed_cache.json` (the same Judge Fast-Path
cache `qknee.ui.dashboard`/`qknee.ui.analysis_app` serve from) for the
showcase: decoding one case's pre-blended, base64-embedded Grad-CAM overlay
costs a base64 decode + PNG decode, nothing more — no ResNet18 forward
pass, no PennyLane QNode execution.

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
CLINICAL_CASE_WALKTHROUGH_PATH = DECK_FIGURES_DIR / "clinical_case_walkthrough.png"

TAGLINE = "NISQ-Ready Variational Quantum Machine Learning for Rapid Orthopedic Triage"

# One representative case per ground-truth category
# (Normal / ACL Tear / Meniscal Tear), picked from `precomputed_cache.json`'s
# 10 pre-scored demo cases, for the landing page's 1-click showcase.
SHOWCASE_CASE_IDS: tuple[str, ...] = ("case_0001", "case_0005", "case_0008")

# Session-state keys shared with `qknee.ui.dashboard` — its `main()` reads
# `VIEW_STATE_KEY` to decide whether to render this landing page or the
# full diagnostic/benchmark console, and this module's CTA buttons write
# to it (then trigger a rerun) to navigate there.
VIEW_STATE_KEY = "qknee_active_view"
VIEW_LANDING = "landing"
VIEW_DIAGNOSTIC = "diagnostic"
VIEW_BENCHMARK = "benchmark"


# --------------------------------------------------------------------------- #
# Data loading — dynamic metric callouts + sample showcase
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def _load_precomputed_cases() -> List[Dict]:
    """Loads `precomputed_cache.json`'s cases list; `[]` (logged) if the
    cache hasn't been built yet (`python scripts/generate_demo_cache.py`)
    or fails to parse, so the showcase can degrade to an explanatory
    message instead of erroring.

    `max_entries=10, ttl=3600`: this function takes no arguments (only
    ever one real entry), but capped consistently with every other raw-
    data-loading cache in this project so a stale read after the on-disk
    file changes doesn't linger indefinitely — see the 1GB-Streamlit-
    Cloud-ceiling budget this module is audited against."""
    if not PRECOMPUTED_CACHE_PATH.exists():
        logger.info("No precomputed cache found at %s; Live Sample Showcase disabled.", PRECOMPUTED_CACHE_PATH)
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
    been generated yet — powers the "Quantum Circuit Latency" hero metric
    with a real measured figure instead of a fabricated one. Same
    `max_entries=10, ttl=3600` capping rationale as `_load_precomputed_cases`
    above."""
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
    bottleneck baseline (same architecture/formula
    `scripts/generate_deck_assets.py`'s efficiency figure uses) — computed
    from the live config, not a hardcoded marketing number, so this stays
    accurate if `config.yaml`'s qubit/layer count ever changes."""
    feature_dim = _config.resnet.feature_dim
    n_qubits = _config.quantum.n_qubits
    linear_params = (feature_dim * n_qubits + n_qubits) + (n_qubits * 2 + 2)
    vqc_params = _config.quantum.n_layers * n_qubits * 3 + n_qubits + 1  # RX/RY/RZ per qubit per layer + readout
    return (1 - vqc_params / linear_params) * 100


def _quantum_latency_ms() -> Optional[float]:
    """Reads the Hybrid Q-Knee VQC's measured per-sample latency from the
    most recent `scripts/run_benchmark.py` export, if one exists. Returns
    `None` (the caller renders a static fallback) rather than a fabricated
    number when no benchmark has been run yet."""
    results = _load_benchmark_results()
    if results is None:
        return None
    for model in results.get("models", []):
        if "VQC" in model.get("name", ""):
            latency = model.get("latency_ms_per_sample")
            return float(latency) if latency is not None else None
    return None


@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def _decode_case_overlay(case: Dict) -> Optional[np.ndarray]:
    """Decodes one precomputed-cache case's base64-embedded, pre-blended
    Grad-CAM overlay PNG (BGR) — the same zero-inference decode
    `qknee.ui.analysis_app.build_fast_path_result` uses. Returns `None`
    (logged) if the case has no embedded heatmap or it fails to decode.
    Cached (`max_entries=10, ttl=3600`) so repeatedly toggling a preview
    open/closed doesn't re-decode the same PNG on every rerun."""
    heatmap_b64 = case.get("heatmap_base64")
    if not heatmap_b64:
        return None
    try:
        import cv2

        png_bytes = base64.b64decode(heatmap_b64)
        return cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001 - showcase degrades gracefully, never crashes the landing page
        logger.warning("Failed to decode showcase heatmap for case %s: %s", case.get("case_id"), exc)
        return None


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

_HERO_CSS = """
<style>
.qknee-hero {
    padding: 2.4rem 2rem 1.8rem 2rem;
    border-radius: 0.9rem;
    background: linear-gradient(135deg, #0B1420 0%, #16222A 55%, #10202B 100%);
    border: 1px solid #22303C;
    margin-bottom: 1.4rem;
    text-align: center;
}
.qknee-hero-eyebrow {
    color: #4FD1C5;
    letter-spacing: 0.12em;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.qknee-hero-title {
    font-size: 3rem;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
    color: #F0F4F8;
}
.qknee-hero-tagline {
    font-size: 1.15rem;
    color: #C5D1DB;
    max-width: 46rem;
    margin: 0 auto;
}
.qknee-hero-disclaimer {
    font-size: 0.75rem;
    color: #8B949E;
    margin-top: 1rem;
}
.qknee-step-figure-caption {
    font-size: 0.82rem;
    color: #8B949E;
    text-align: center;
}
</style>
"""


# --------------------------------------------------------------------------- #
# 1. Hero & branding
# --------------------------------------------------------------------------- #

def render_hero() -> None:
    """Title, tagline, three dynamic quick-metric callout cards, and the
    two primary CTA buttons."""
    st.markdown(_HERO_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="qknee-hero">
            <div class="qknee-hero-eyebrow">QUANTUM-ASSISTED ORTHOPEDIC MRI TRIAGE</div>
            <div class="qknee-hero-title">🦵 Q-Knee</div>
            <div class="qknee-hero-tagline">{TAGLINE}</div>
            <div class="qknee-hero-disclaimer">
                Research prototype — not a certified medical device. Not for clinical use.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_qubits = _config.quantum.n_qubits
    n_layers = _config.quantum.n_layers
    reduction_pct = _parameter_reduction_pct()
    latency_ms = _quantum_latency_ms()

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        with st.container(border=True):
            st.metric(
                "Parameter Reduction",
                f"{reduction_pct:.1f}%",
                help=(
                    f"The {n_qubits}-qubit VQC head's trainable parameters vs. an equivalent classical "
                    f"linear bottleneck head of the same input/output width, computed live from the "
                    f"current config (qknee.config.config.yaml)."
                ),
            )
    with metric_col2:
        with st.container(border=True):
            if latency_ms is not None:
                st.metric(
                    "Quantum Circuit Latency",
                    f"{latency_ms:.1f} ms",
                    help="Measured per-sample VQC inference latency from the most recent "
                         "`python scripts/run_benchmark.py` run.",
                )
            else:
                st.metric(
                    "Quantum Circuit Latency",
                    "NISQ Sim",
                    help="Run `python scripts/run_benchmark.py` to measure and display a live "
                         "per-sample latency figure here.",
                )
    with metric_col3:
        with st.container(border=True):
            st.metric(
                "Variational Circuit",
                f"{n_qubits}-Qubit PQC",
                help=f"{n_layers}-layer parameterized quantum circuit — angle-encoded, ring-entangled, "
                     "Pauli-Z read out.",
            )

    st.write("")
    cta_col1, cta_col2 = st.columns(2)
    with cta_col1:
        if st.button("🚀 Launch Live Diagnostic Console", type="primary", use_container_width=True):
            st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
            st.rerun()
    with cta_col2:
        if st.button("📊 Explore Clinical Benchmarks", use_container_width=True):
            st.session_state[VIEW_STATE_KEY] = VIEW_BENCHMARK
            st.rerun()


# --------------------------------------------------------------------------- #
# 2. Product & architecture showcase — interactive 3-step pipeline explainer
# --------------------------------------------------------------------------- #

def render_pipeline_explainer() -> None:
    """Three-tab interactive walkthrough of the ResNet18 -> quantum circuit
    -> Grad-CAM/report pipeline, with the offline-rendered circuit diagram
    and clinical-case-walkthrough figures (`scripts/generate_deck_assets.py`)
    embedded inline."""
    st.markdown("## 🧭 How Q-Knee Works")
    st.caption(
        "Three stages, one forward pass: a classical vision backbone compresses each MRI slice, a "
        "4-qubit variational circuit scores tear risk, and Grad-CAM plus an auto-generated PDF explain why."
    )

    n_qubits = _config.quantum.n_qubits
    n_layers = _config.quantum.n_layers
    feature_dim = _config.resnet.feature_dim

    step1_tab, step2_tab, step3_tab = st.tabs([
        "① Spatial Vision Backbone",
        "② Quantum Circuit Execution",
        "③ Explainability & Report",
    ])

    with step1_tab:
        text_col, fig_col = st.columns([1.1, 1])
        with text_col:
            st.markdown(
                f"""
**ResNet18 feature extraction.** Each MRI slice (or an aggregated
multi-slice volume) is passed through a frozen, ImageNet-pretrained
ResNet18 backbone, producing a dense **{feature_dim}-dimensional** feature
vector — the same spatial-vision embedding every downstream stage
consumes (`qknee.models.resnet_extractor.ResNet18FeatureExtractor`).

- Frozen convolutional backbone — no retraining required
- {feature_dim}-D embedding per slice
- Multi-slice volumes aggregate via mean / attention / top-k pooling
                """
            )
        with fig_col:
            st.info(f"**Slice → ResNet18 → {feature_dim}-D embedding**", icon="🩻")
            st.caption("The classical stage every quantum stage below builds on.")

    with step2_tab:
        text_col, fig_col = st.columns([1, 1.25])
        with text_col:
            st.markdown(
                f"""
**Continuous-variable angle encoding + {n_qubits}-qubit entangled circuit.**
A classical PCA projection compresses the {feature_dim}-D embedding down to
**{n_qubits} continuous angles** in `[0, 2π]`, one per qubit. Each angle is
angle-encoded via `RX`→`RY` rotations, followed by **{n_layers} trainable
variational layers** (per-qubit `RX`/`RY`/`RZ` rotations + a ring of `CNOT`
entangling gates), then every qubit is measured in the Pauli-Z basis.

- {feature_dim}-D → {n_qubits}-D PCA compression, mapped to `[0, 2π]`
- {n_layers}-layer parameterized quantum circuit (PennyLane `default.qubit`)
- Ring `CNOT` entanglement, Pauli-Z `⟨Z⟩` measurement
                """
            )
        with fig_col:
            if CIRCUIT_DIAGRAM_PATH.exists():
                st.image(str(CIRCUIT_DIAGRAM_PATH), caption="4-Qubit Variational Circuit — Q-Knee", use_container_width=True)
            else:
                st.warning(
                    "Circuit diagram not yet generated. Run `python scripts/generate_deck_assets.py`.",
                    icon="⚠️",
                )

    with step3_tab:
        st.markdown(
            """
**Explainable Grad-CAM + automated radiology report.** Grad-CAM highlights
which spatial regions of the slice drove the ResNet18 embedding, the VQC's
per-qubit Pauli-Z measurements show the quantum circuit's own attribution,
and `qknee.xai.report_generator.ReportGenerator` compiles the slice, the
Grad-CAM overlay, and the full quantum/clinical attribution breakdown into
a structured, downloadable PDF report.
            """
        )
        if CLINICAL_CASE_WALKTHROUGH_PATH.exists():
            st.image(
                str(CLINICAL_CASE_WALKTHROUGH_PATH),
                caption="Clinical Case Walkthrough — Slice to Diagnosis",
                use_container_width=True,
            )
        else:
            st.warning(
                "Clinical case walkthrough figure not yet generated. Run `python scripts/generate_deck_assets.py`.",
                icon="⚠️",
            )


# --------------------------------------------------------------------------- #
# 3. Interactive live sample showcase — 1-click preview, no login required
# --------------------------------------------------------------------------- #

def render_live_sample_showcase() -> None:
    """Lets a visitor preview 3 pre-scored knee MRI cases (one Normal, one
    ACL Tear, one Meniscal Tear) with a single click, straight from
    `precomputed_cache.json` — no login, no upload, no live inference."""
    st.markdown("## 🔎 Try It Now — Live Sample Cases")
    st.caption(
        "Preview 3 pre-scored knee MRI cases straight from the precomputed inference cache — "
        "zero login, zero upload, zero wait."
    )

    all_cases = _load_precomputed_cases()
    by_id = {case["case_id"]: case for case in all_cases}
    showcase_cases = [by_id[case_id] for case_id in SHOWCASE_CASE_IDS if case_id in by_id]

    if not showcase_cases:
        st.info(
            "No precomputed sample cases found yet. Run `python scripts/generate_demo_cache.py` "
            "to build the showcase cache.",
            icon="ℹ️",
        )
        return

    columns = st.columns(len(showcase_cases))
    for column, case in zip(columns, showcase_cases):
        with column:
            with st.container(border=True):
                st.markdown(f"**{case.get('ground_truth_category', 'Unknown')}**")
                st.caption(f"{case.get('plane', '?').capitalize()} plane · {case['case_id']}")

                preview_key = f"_qknee_showcase_preview_{case['case_id']}"
                if st.button(
                    "▶️ Preview this case", key=f"qknee_showcase_btn_{case['case_id']}", use_container_width=True,
                ):
                    st.session_state[preview_key] = True

                if st.session_state.get(preview_key):
                    overlay_bgr = _decode_case_overlay(case)
                    if overlay_bgr is not None:
                        st.image(
                            overlay_bgr[:, :, ::-1],  # BGR -> RGB for st.image
                            caption="Grad-CAM risk overlay",
                            use_container_width=True,
                        )
                    else:
                        st.warning("Heatmap unavailable for this case.", icon="⚠️")

                    risk_score = float(case.get("risk_score", 0.0))
                    st.metric(
                        "Tear Risk",
                        f"{risk_score * 100:.1f}%",
                        delta=case.get("risk_tier", "N/A"),
                        delta_color="inverse" if case.get("risk_tier") == "LOW" else "off",
                    )
                    st.caption(case.get("classification_label", ""))

    st.caption(
        f"Every scan above was produced by the real Q-Knee pipeline "
        f"(`scripts/generate_demo_cache.py`), then cached to "
        f"`{PRECOMPUTED_CACHE_PATH.relative_to(_REPO_ROOT)}` for instant, zero-latency replay here."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def render_landing_page() -> None:
    """Renders the full public landing page: hero + CTAs, the 3-step
    pipeline explainer, and the live sample showcase. Call this from
    `qknee.ui.dashboard.main()` when `st.session_state[VIEW_STATE_KEY]`
    is `VIEW_LANDING` (its default)."""
    render_hero()
    st.divider()
    render_pipeline_explainer()
    st.divider()
    render_live_sample_showcase()
