"""
Q-Knee public landing page (Streamlit) — the marketing/onboarding entry
view wired into `qknee.ui.dashboard`'s `main()`. Refactored to strictly
mirror the project's PRD: the clinical problem statement (radiologist
backlog on high-volume ACL/meniscal-tear triage), the 5-stage hybrid
quantum-classical execution pipeline (Section 3 of the PRD), the 3
technical pillars, and the Plan-B precomputed-cache cohort roster used
for zero-latency 1-click case evaluation.

Reuses `qknee/artifacts/precomputed_cache.json` (the same Judge Fast-Path
cache `qknee.ui.dashboard`/`qknee.ui.analysis_app` serve from) for the
cohort roster: decoding one case's pre-blended, base64-embedded Grad-CAM
overlay costs a base64 decode + PNG decode, nothing more — no ResNet18
forward pass, no PennyLane QNode execution.

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
CLINICAL_CASE_WALKTHROUGH_PATH = DECK_FIGURES_DIR / "clinical_case_walkthrough.png"

TAGLINE = "NISQ-Ready Hybrid Quantum ML for Knee Abnormality Triage"
AUTHOR_CREDIT = "Yashika Nayak"

# P0 MVP cohort (PRD Section 6 / "Plan B" latency-mitigation cache): one
# ACL tear (Sagittal), one medial meniscus tear (Coronal), one normal
# control (Sagittal) — picked from `precomputed_cache.json`'s 10 pre-scored
# demo cases by matching category+plane. `CASE_DISPLAY_UID` renders the
# exact case numbers called out in the PRD (#0428 / #0112 / #0093) without
# renaming the underlying cache files those cases actually live in.
SHOWCASE_CASE_IDS: tuple[str, ...] = ("case_0005", "case_0008", "case_0001")
CASE_DISPLAY_UID: Dict[str, str] = {
    "case_0005": "0428",
    "case_0008": "0112",
    "case_0001": "0093",
}

# Session-state keys shared with `qknee.ui.dashboard` — its `main()` reads
# `VIEW_STATE_KEY` to decide whether to render this landing page or the
# full diagnostic/benchmark console, and this module's CTA buttons write
# to it (then trigger a rerun) to navigate there.
VIEW_STATE_KEY = "qknee_active_view"
VIEW_LANDING = "landing"
VIEW_DIAGNOSTIC = "diagnostic"
VIEW_BENCHMARK = "benchmark"


# --------------------------------------------------------------------------- #
# Data loading — dynamic metric callouts + cohort roster
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False, max_entries=10, ttl=3600)
def _load_precomputed_cases() -> List[Dict]:
    """Loads `precomputed_cache.json`'s cases list; `[]` (logged) if the
    cache hasn't been built yet (`python scripts/generate_demo_cache.py`)
    or fails to parse, so the cohort roster can degrade to an explanatory
    message instead of erroring.

    `max_entries=10, ttl=3600`: this function takes no arguments (only
    ever one real entry), but capped consistently with every other raw-
    data-loading cache in this project so a stale read after the on-disk
    file changes doesn't linger indefinitely — see the 1GB-Streamlit-
    Cloud-ceiling budget this module is audited against."""
    if not PRECOMPUTED_CACHE_PATH.exists():
        logger.info("No precomputed cache found at %s; Cohort Evaluation Roster disabled.", PRECOMPUTED_CACHE_PATH)
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
    been generated yet — powers the hero's measured-latency callout with
    a real figure instead of a fabricated one. Same `max_entries=10,
    ttl=3600` capping rationale as `_load_precomputed_cases` above."""
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
# Styling — Q-Knee's clinical-portal design language (Section 1 of the
# PRD): sterile off-white canvas, pure white cards, 1px slate borders,
# deep clinical-slate typography, surgical emerald + diagnostic blue
# accents. Extends `qknee.ui.theme`'s shared tokens with rules specific to
# this page; `qknee.ui.dashboard`'s workstation/benchmark tabs keep the
# same sterile clinical card/badge system via `theme.inject_clinical_theme()`.
# --------------------------------------------------------------------------- #

# Local palette — the PRD's exact clinical-portal spec (Section 1), kept
# independent of `qknee.ui.theme`'s shared tokens (which this page's edit
# scope does not include) so this page mirrors the PRD precisely
# regardless of how the shared theme evolves elsewhere in the app.
LP_BG = "#F8FAFC"              # sterile off-white canvas
LP_CARD = "#FFFFFF"            # pure white cards
LP_BORDER = "#E2E8F0"          # subtle 1px border
LP_BORDER_STRONG = "#CBD5E1"
LP_SLATE = "#0F172A"           # deep clinical slate typography
LP_MUTED = "#475569"
LP_MONO = "#1E293B"
LP_EMERALD = "#0D9488"         # surgical emerald — primary accent
LP_BLUE = "#0284C7"            # diagnostic blue — secondary accent
LP_RADIUS = "8px"
LP_SHADOW = "0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04)"

_LANDING_CSS = f"""
<style>
/* Command bar */
.qknee-command-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
    padding: 0.9rem 1.5rem;
    background: {LP_CARD};
    border: 1px solid {LP_BORDER};
    border-radius: {LP_RADIUS};
    box-shadow: {LP_SHADOW};
    margin-bottom: 0;
}}
.qknee-command-brand {{
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: -0.005em;
    color: {LP_SLATE};
}}
.qknee-command-brand .qknee-brand-accent {{ color: {LP_EMERALD}; }}
.qknee-command-subtitle {{
    font-size: 0.68rem;
    color: {LP_MUTED};
    letter-spacing: 0.03em;
    margin-top: 0.15rem;
}}
.qknee-telemetry-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}}
.qknee-telemetry-chip {{
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
    font-size: 0.63rem;
    font-weight: 600;
    color: {LP_MONO};
    background: {LP_BG};
    border: 1px solid {LP_BORDER};
    border-radius: {LP_RADIUS};
    padding: 0.22rem 0.55rem;
    white-space: nowrap;
}}

/* Hero */
.qknee-hero {{
    padding: 2.6rem 2.4rem;
    border-radius: {LP_RADIUS};
    background: {LP_CARD};
    border: 1px solid {LP_BORDER};
    box-shadow: {LP_SHADOW};
    margin: 1.2rem 0 0 0;
}}
.qknee-hero-eyebrow {{
    color: {LP_EMERALD};
    background: {LP_EMERALD}12;
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: {LP_RADIUS};
    letter-spacing: 0.08em;
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 1rem;
}}
.qknee-hero-title {{
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -0.01em;
    margin: 0 0 1rem 0;
    color: {LP_SLATE};
    max-width: 40rem;
}}
.qknee-hero-problem {{
    font-size: 0.95rem;
    color: {LP_MUTED};
    line-height: 1.7;
    max-width: 46rem;
}}
.qknee-hero-problem b {{ color: {LP_SLATE}; }}

/* Section heading (centered) */
.qknee-section-heading {{
    text-align: center;
    margin: 0.5rem 0 2rem 0;
}}
.qknee-section-heading h2 {{
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin-bottom: 0.4rem;
}}
.qknee-section-heading p {{
    font-size: 0.88rem;
    color: {LP_MUTED};
    max-width: 40rem;
    margin: 0 auto;
}}

/* 5-stage pipeline cards */
.qknee-stage-card {{
    background: {LP_CARD};
    border: 1px solid {LP_BORDER};
    border-radius: {LP_RADIUS};
    box-shadow: {LP_SHADOW};
    padding: 1.3rem 1.05rem;
    height: 100%;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}}
.qknee-stage-card:hover {{
    box-shadow: 0 10px 24px rgba(2, 132, 199, 0.12);
    transform: translateY(-2px);
}}
.qknee-stage-index {{
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: {LP_EMERALD};
    margin-bottom: 0.5rem;
}}
.qknee-stage-title {{
    font-size: 0.87rem;
    font-weight: 700;
    color: {LP_SLATE};
    margin-bottom: 0.45rem;
    line-height: 1.3;
}}
.qknee-stage-body {{
    font-size: 0.76rem;
    color: {LP_MUTED};
    line-height: 1.55;
}}
.qknee-stage-arrow {{
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    font-size: 1.2rem;
    color: {LP_BORDER_STRONG};
}}

/* 3 technical pillars */
.qknee-pillar-card {{
    background: {LP_CARD};
    border: 1px solid {LP_BORDER};
    border-radius: {LP_RADIUS};
    box-shadow: {LP_SHADOW};
    padding: 1.5rem 1.3rem;
    height: 100%;
}}
.qknee-pillar-eyebrow {{
    font-family: 'JetBrains Mono', 'SF Mono', Consolas, ui-monospace, monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: {LP_EMERALD};
    margin-bottom: 0.5rem;
}}
.qknee-pillar-title {{
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}
.qknee-pillar-body {{
    font-size: 0.82rem;
    color: {LP_MUTED};
    line-height: 1.6;
}}

/* Footer */
.qknee-footer {{
    background: {LP_SLATE};
    color: rgba(255,255,255,0.85);
    border-radius: {LP_RADIUS};
    padding: 1.8rem 2rem 1.2rem 2rem;
    margin-top: 0.5rem;
}}
.qknee-footer h4 {{ color: #FFFFFF; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.6rem; }}
.qknee-footer p {{ font-size: 0.78rem; color: rgba(255,255,255,0.72); line-height: 1.7; }}
.qknee-footer-disclaimer {{
    border-top: 1px solid rgba(255,255,255,0.15);
    margin-top: 1.2rem;
    padding-top: 0.9rem;
    font-size: 0.72rem;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    text-align: center;
}}
.qknee-footer-credits {{
    font-size: 0.68rem;
    color: rgba(255,255,255,0.55);
    text-align: center;
    margin-top: 0.5rem;
}}

.qknee-step-figure-caption {{
    font-size: 0.8rem;
    color: {LP_MUTED};
    text-align: center;
}}
</style>
"""

TELEMETRY_ITEMS: tuple[str, ...] = (
    "DATASET: Stanford MRNet / RSNA Knee MRI",
    "BACKEND: PennyLane 4-Qubit VQC (Statevector)",
    "STATUS: NISQ Simulator Online",
)


def _inject_landing_css() -> None:
    st.markdown(_LANDING_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# 1. Header & Command Bar
# --------------------------------------------------------------------------- #

def render_command_bar() -> None:
    """Institutional command bar: brand identity + authorship on the left,
    live clinical-telemetry chips in the center, and a high-contrast
    `Launch Diagnostic Workstation` action on the right. This is the
    marketing page's own header — the authenticated workstation keeps its
    separate functional navbar, rendered globally by
    `qknee.ui.auth_view.render_global_navbar`."""
    _inject_landing_css()
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)

    brand_col, telemetry_col, action_col = st.columns([1.6, 2.6, 1.5])
    with brand_col:
        st.markdown(
            f"""
            <div class="qknee-command-bar" style="border-right:none;">
                <div>
                    <div class="qknee-command-brand">Q-KNEE <span class="qknee-brand-accent">&bull;</span></div>
                    <div class="qknee-command-subtitle">
                        Musculoskeletal Radiology Research Suite &middot; Author: {AUTHOR_CREDIT}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with telemetry_col:
        chips = "".join(f'<span class="qknee-telemetry-chip">{item}</span>' for item in TELEMETRY_ITEMS)
        st.markdown(
            f'<div class="qknee-command-bar" style="border-left:none; border-right:none;">'
            f'<div class="qknee-telemetry-row">{chips}</div></div>',
            unsafe_allow_html=True,
        )
    with action_col:
        st.markdown('<div style="height:0.55rem;"></div>', unsafe_allow_html=True)
        if st.button("Launch Diagnostic Workstation →", type="primary", use_container_width=True,
                     key="qknee_command_launch"):
            st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
            st.rerun()


# --------------------------------------------------------------------------- #
# 2. Hero — clinical vision & problem statement
# --------------------------------------------------------------------------- #

def render_hero() -> None:
    """Hero block stating the clinical problem this project solves
    (radiologist backlog on high-volume ACL/meniscal triage, and why
    classical deep learning alone struggles with it) and the two primary
    CTAs the PRD specifies."""
    st.markdown(
        f"""
        <div class="qknee-hero">
            <div class="qknee-hero-eyebrow">AI &amp; Quantum Innovation // 24-Hour Sprint Benchmark</div>
            <div class="qknee-hero-title">Accessible Hybrid Quantum AI for High-Volume Knee MRI Triage</div>
            <div class="qknee-hero-problem">
                Emergency and sports-medicine radiology departments face a growing backlog of knee MRI
                studies requiring slice-by-slice 3D volumetric review to catch Anterior Cruciate Ligament
                (ACL) and meniscal tears — a labor-intensive read that doesn't scale with case volume.
                <b>Classical deep learning models struggle here</b>: dense classification heads over a
                high-dimensional feature space are compute-hungry and can miss the non-linear correlations
                across slices that distinguish a tear from a normal variant. Q-Knee offloads that non-linear
                boundary learning to a <b>quantum Hilbert space</b> — a compact, angle-encoded 4-qubit
                variational circuit — without requiring fault-tolerant, multi-million-dollar quantum hardware:
                it runs today on a NISQ simulator.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    cta_col1, cta_col2 = st.columns(2)
    with cta_col1:
        if st.button("Open Diagnostic Workstation", type="primary", use_container_width=True, key="qknee_hero_cta"):
            st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
            st.rerun()
    with cta_col2:
        st.markdown(
            f"""
            <a href="#pipeline" style="text-decoration:none; display:block;">
                <div style="text-align:center; padding:0.5rem 0; font-weight:700; border:1px solid
                    {LP_BORDER_STRONG}; border-radius:999px; color:{LP_SLATE};">
                    View 5-Stage Technical Pipeline
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# 3. 5-Stage Hybrid Quantum-Classical Execution Pipeline (PRD Section 3)
# --------------------------------------------------------------------------- #

def render_pipeline_stages() -> None:
    """Interactive 5-column technical pipeline diagram matching PRD
    Section 3 exactly: ingestion -> feature extraction -> compression ->
    quantum circuit -> inference/XAI."""
    st.markdown('<div id="pipeline"></div>', unsafe_allow_html=True)
    feature_dim = _config.resnet.feature_dim
    n_qubits = _config.quantum.n_qubits

    st.markdown(
        """
        <div class="qknee-section-heading">
            <h2>5-Stage Hybrid Quantum-Classical Execution Pipeline</h2>
            <p>One forward pass, five stages — from a raw DICOM/NPY slice to a Pauli-Z risk score with
               a Grad-CAM saliency map a radiologist can verify at a glance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stages = [
        ("01", "DICOM/NPY Ingestion", "Volume normalization &amp; slice resizing (128&times;128) via "
         "<code>SimpleITK</code> / <code>pydicom</code>."),
        ("02", "Spatial Feature Extraction", f"Frozen <code>ResNet18</code> backbone projecting slice "
         f"volumes into a {feature_dim}-dimensional continuous latent space."),
        ("03", "Feature Compression", f"Linear bottleneck / PCA compressing vectors from {feature_dim}-dim "
         f"down to {n_qubits} quantum-ready continuous scalars."),
        ("04", "Parameterized Quantum Circuit", f"Angle encoding onto {n_qubits} qubits with parameterized "
         "rotational (R<sub>X</sub>, R<sub>Z</sub>) and entangling CNOT gates in <code>PennyLane</code> "
         "(<code>TorchLayer</code>)."),
        ("05", "Inference &amp; XAI", "Pauli-Z expectation value risk scoring paired with <code>Grad-CAM</code> "
         "visual saliency maps for radiologist verification."),
    ]

    cols = st.columns([2, 0.3, 2, 0.3, 2, 0.3, 2, 0.3, 2])
    stage_cols = [cols[0], cols[2], cols[4], cols[6], cols[8]]
    arrow_cols = [cols[1], cols[3], cols[5], cols[7]]

    for col, (index, title, body) in zip(stage_cols, stages):
        with col:
            st.markdown(
                f"""
                <div class="qknee-stage-card">
                    <div class="qknee-stage-index">{index}.</div>
                    <div class="qknee-stage-title">{title}</div>
                    <div class="qknee-stage-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    for col in arrow_cols:
        with col:
            st.markdown('<div class="qknee-stage-arrow">&rarr;</div>', unsafe_allow_html=True)

    with st.expander("Deep-Dive: Full Pipeline Walkthrough (circuit diagram + case walkthrough)"):
        render_pipeline_deep_dive()


def render_pipeline_deep_dive() -> None:
    """Detailed 3-tab walkthrough (feature extraction / quantum circuit /
    attribution+report) with the offline-rendered circuit diagram and
    clinical-case-walkthrough figures (`scripts/generate_deck_assets.py`)
    embedded inline — nested under the 5-stage summary above rather than
    a separate top-level section, so it stays additive detail on the same
    PRD-defined pipeline rather than a duplicate section."""
    n_qubits = _config.quantum.n_qubits
    n_layers = _config.quantum.n_layers
    feature_dim = _config.resnet.feature_dim

    step1_tab, step2_tab, step3_tab = st.tabs([
        "Stage 01-02 — Ingestion & Feature Extraction",
        "Stage 03-04 — Compression & Quantum Circuit",
        "Stage 05 — Inference & XAI",
    ])

    with step1_tab:
        text_col, fig_col = st.columns([1.1, 1])
        with text_col:
            st.markdown(
                f"""
**DICOM/NPY ingestion + ResNet18 feature extraction.** Each MRI slice (or an
aggregated multi-slice volume) is normalized and resized to 128&times;128,
then passed through a frozen, ImageNet-pretrained ResNet18 backbone,
producing a dense **{feature_dim}-dimensional** feature vector — the same
radiological embedding every downstream stage consumes
(`qknee.models.resnet_extractor.ResNet18FeatureExtractor`).

- Volume normalization &amp; slice resizing via SimpleITK / pydicom
- Frozen convolutional backbone — no retraining required
- {feature_dim}-D embedding per slice
                """
            )
        with fig_col:
            st.info(f"Slice → ResNet18 → {feature_dim}-D Embedding")
            st.caption("The classical stage every quantum stage below builds on.")

    with step2_tab:
        text_col, fig_col = st.columns([1, 1.25])
        with text_col:
            st.markdown(
                f"""
**Feature compression + {n_qubits}-qubit entangled circuit.** A linear
bottleneck / PCA projection compresses the {feature_dim}-D embedding down to
**{n_qubits} continuous angles** in `[0, 2π]`, one per qubit. Each angle is
angle-encoded via `RX`→`RZ` rotations, followed by **{n_layers} trainable
variational layers** (per-qubit `RX`/`RY`/`RZ` rotations + a ring of `CNOT`
entangling gates), then every qubit is measured in the Pauli-Z basis.

- {feature_dim}-D → {n_qubits}-D compression, mapped to `[0, 2π]`
- {n_layers}-layer parameterized quantum circuit (PennyLane `default.qubit`, `TorchLayer`)
- Ring `CNOT` entanglement, Pauli-Z `⟨Z⟩` measurement
                """
            )
        with fig_col:
            if CIRCUIT_DIAGRAM_PATH.exists():
                st.image(str(CIRCUIT_DIAGRAM_PATH), caption="4-Qubit Variational Circuit — Q-Knee", use_container_width=True)
            else:
                st.warning("Circuit diagram not yet generated. Run `python scripts/generate_deck_assets.py`.")

    with step3_tab:
        st.markdown(
            """
**Pauli-Z risk scoring + Grad-CAM XAI.** Grad-CAM performs quantitative
lesion localization, highlighting which spatial regions of the slice drove
the ResNet18 embedding; the VQC's per-qubit Pauli-Z measurements supply a
formal attribution breakdown of the quantum circuit's own contribution;
and `qknee.xai.report_generator` compiles the slice, the Grad-CAM overlay,
and the full quantum/clinical attribution breakdown into a structured,
downloadable diagnostic PDF report for radiologist verification.
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
                "Clinical case walkthrough figure not yet generated. Run `python scripts/generate_deck_assets.py`."
            )


# --------------------------------------------------------------------------- #
# 4. Technical Specification & Innovation Grid (3 Key Pillars)
# --------------------------------------------------------------------------- #

def render_technical_pillars() -> None:
    """The PRD's 3 technical pillars — hardware-efficient quantum
    advantage, explainable AI for radiological trust, and multi-plane
    volumetric triage — replacing any generic feature-card content."""
    reduction_pct = _parameter_reduction_pct()

    st.markdown(
        """
        <div class="qknee-section-heading">
            <h2>Technical Specification &amp; Innovation</h2>
            <p>Three pillars underpin the Q-Knee architecture.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pillars = [
        ("Pillar 1", "Hardware-Efficient Quantum Advantage",
         f"Evaluates inter-feature correlations using superposition and entanglement on NISQ simulators; "
         f"achieves a {reduction_pct:.0f}% trainable-parameter reduction versus a classical dense head of "
         f"equivalent input dimensionality."),
        ("Pillar 2", "Explainable AI for Radiological Trust",
         "Layer-4 Grad-CAM backpropagation overlaid directly on DICOM slices, pinpointing exact anatomical "
         "regions of interest (ROI) alongside Pauli-Z expectation readouts."),
        ("Pillar 3", "Multi-Plane Volumetric Triage",
         "Evaluates tear probability across Sagittal and Coronal MRI series from the Stanford MRNet cohort."),
    ]
    cols = st.columns(3)
    for col, (eyebrow, title, body) in zip(cols, pillars):
        with col:
            st.markdown(
                f"""
                <div class="qknee-pillar-card">
                    <div class="qknee-pillar-eyebrow">{eyebrow}</div>
                    <div class="qknee-pillar-title">{title}</div>
                    <div class="qknee-pillar-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# 5. One-Click Cohort Evaluation Roster (P0 MVP cases)
# --------------------------------------------------------------------------- #

def _case_display_uid(case: Dict) -> str:
    return CASE_DISPLAY_UID.get(case.get("case_id", ""), str(case.get("case_id", "0000")))


def render_cohort_roster() -> None:
    """Clinical evaluation table with 1-click loading for the PRD's P0 MVP
    cohort (Plan B latency mitigation): Case #0428 (ACL, Sagittal), Case
    #0112 (Medial Meniscus, Coronal), Case #0093 (Normal, Sagittal) —
    served straight from `precomputed_cache.json`, zero inference
    latency. Each `Load & Inspect` opens the case's Quantum Risk Score and
    Grad-CAM overlay inline, plus a route into the Diagnostic Workstation."""
    st.markdown('<div id="cohort"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="qknee-section-heading">
            <h2>One-Click Cohort Evaluation Roster</h2>
            <p>P0 MVP test cases, pre-scored offline and cached for zero-latency 1-click review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    all_cases = _load_precomputed_cases()
    by_id = {case["case_id"]: case for case in all_cases}
    roster_cases = [by_id[case_id] for case_id in SHOWCASE_CASE_IDS if case_id in by_id]

    if not roster_cases:
        st.info(
            "No precomputed sample cases found yet. Run `python scripts/generate_demo_cache.py` "
            "to build the cohort cache."
        )
        return

    import pandas as pd

    table_rows = [
        {
            "Case UID": f"#{_case_display_uid(case)}",
            "Series Plane": str(case.get("plane", "?")).capitalize(),
            "Suspected Pathology": _clinical_indication(case),
            "Quantum Risk Score": f"{float(case.get('risk_score', 0.0)) * 100:.1f}%",
            "Action": "Load & Inspect",
        }
        for case in roster_cases
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.write("")

    columns = st.columns(len(roster_cases))
    for column, case in zip(columns, roster_cases):
        with column:
            with st.container(border=True):
                st.markdown(f"**Case #{_case_display_uid(case)}**")
                st.caption(f"{str(case.get('plane', '?')).capitalize()} · {_clinical_indication(case)}")

                preview_key = f"_qknee_cohort_preview_{case['case_id']}"
                if st.button(
                    "Load & Inspect", key=f"qknee_cohort_btn_{case['case_id']}", use_container_width=True,
                ):
                    st.session_state[preview_key] = True

                if st.session_state.get(preview_key):
                    overlay_bgr = _decode_case_overlay(case)
                    if overlay_bgr is not None:
                        st.image(
                            overlay_bgr[:, :, ::-1],  # BGR -> RGB for st.image
                            caption="Grad-CAM Saliency Overlay",
                            use_container_width=True,
                        )
                    else:
                        st.warning("Attribution overlay unavailable for this case.")

                    risk_score = float(case.get("risk_score", 0.0))
                    st.metric(
                        "Quantum Risk Score",
                        f"{risk_score * 100:.1f}%",
                        delta=case.get("risk_tier", "N/A"),
                        delta_color="inverse" if case.get("risk_tier") == "LOW" else "off",
                    )

                    if st.button(
                        "Open in Diagnostic Workstation", key=f"qknee_cohort_launch_{case['case_id']}",
                        type="primary", use_container_width=True,
                    ):
                        st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
                        st.rerun()

    st.caption(
        f"Every case above was produced by the real Q-Knee pipeline "
        f"(`scripts/generate_demo_cache.py`), then cached to "
        f"`{PRECOMPUTED_CACHE_PATH.relative_to(_REPO_ROOT)}` for instant, zero-latency replay here."
    )


_INDICATION_BY_CATEGORY = {
    "normal": "Normal Control Knee",
    "acl tear": "Anterior Cruciate Ligament (ACL) Tear",
    "meniscal tear": "Medial Meniscus Tear",
    "mcl sprain": "Medial Collateral Ligament (MCL) Sprain",
}


def _clinical_indication(case: Dict) -> str:
    category = str(case.get("ground_truth_category", "")).strip().lower()
    return _INDICATION_BY_CATEGORY.get(category, case.get("ground_truth_category", "Finding"))


# --------------------------------------------------------------------------- #
# 6. Footer & regulatory compliance
# --------------------------------------------------------------------------- #

def render_footer() -> None:
    st.markdown('<div class="qknee-footer">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <h4>Q-Knee Diagnostic Platform</h4>
            <p>{TAGLINE}<br>Musculoskeletal Radiology Research Suite<br>Author: {AUTHOR_CREDIT}</p>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <h4>Tech Stack</h4>
            <p>Built with PyTorch 2.x &bull; PennyLane &bull; Qiskit Aer &bull; SimpleITK &bull;
               MONAI &bull; Streamlit Cloud</p>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <div class="qknee-footer-disclaimer">
            INVESTIGATIONAL PROTOTYPE ONLY &bull; RSNA / Stanford MRNet Validation &bull;
            Confirmatory over-read required by a board-certified radiologist.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="qknee-footer-credits">© Q-Knee Research Prototype</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def render_landing_page() -> None:
    """Renders the full PRD-aligned public landing page: command bar,
    clinical-problem hero, 5-stage execution pipeline, 3 technical
    pillars, the one-click cohort evaluation roster, and the regulatory
    footer. Call this from `qknee.ui.dashboard.main()` when
    `st.session_state[VIEW_STATE_KEY]` is `VIEW_LANDING` (its default)."""
    render_command_bar()
    render_hero()
    st.write("")
    render_pipeline_stages()
    st.divider()
    render_technical_pillars()
    st.divider()
    render_cohort_roster()
    st.write("")
    render_footer()
