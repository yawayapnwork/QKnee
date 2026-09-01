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
from qknee.ui import theme

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
# Styling — extends `qknee.ui.theme`'s shared clinical design system with
# a handful of hero-section-only rules (the rest of the page reuses
# `.qknee-card`/`.qknee-badge`/etc. directly).
# --------------------------------------------------------------------------- #

_HERO_CSS = f"""
<style>
.qknee-hero {{
    padding: 2.2rem 2rem 1.7rem 2rem;
    border-radius: 0.7rem;
    background: linear-gradient(135deg, #0B1420 0%, #111C33 55%, #0D1B2A 100%);
    border: 1px solid {theme.BORDER_GREY}30;
    margin-bottom: 1.2rem;
    text-align: center;
}}
.qknee-hero-eyebrow {{
    color: {theme.SURGICAL_TEAL};
    letter-spacing: 0.12em;
    font-size: 0.76rem;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}
.qknee-hero-title {{
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin: 0 0 0.5rem 0;
    color: {theme.STERILE_WHITE};
}}
.qknee-hero-tagline {{
    font-size: 1.05rem;
    color: {theme.TEXT_MUTED};
    max-width: 46rem;
    margin: 0 auto;
}}
.qknee-step-figure-caption {{
    font-size: 0.8rem;
    color: {theme.TEXT_MUTED};
    text-align: center;
}}
</style>
"""


# --------------------------------------------------------------------------- #
# 1. Hero & branding
# --------------------------------------------------------------------------- #

def render_hero() -> None:
    """Institutional header (laboratory branding + system status), the
    NISQ Clinical Research Disclosure banner, the hero title/tagline,
    three formal capability-summary cards, and the two primary CTA
    buttons."""
    theme.inject_clinical_theme()
    theme.render_institutional_masthead(active_module="Institutional Overview")
    theme.render_disclosure_banner()

    st.markdown(_HERO_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="qknee-hero">
            <div class="qknee-hero-eyebrow">Quantum-Assisted Orthopedic MRI Triage</div>
            <div class="qknee-hero-title">Q-KNEE Diagnostic Workstation</div>
            <div class="qknee-hero-tagline">{TAGLINE}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_qubits = _config.quantum.n_qubits
    n_layers = _config.quantum.n_layers
    feature_dim = _config.resnet.feature_dim
    reduction_pct = _parameter_reduction_pct()
    latency_ms = _quantum_latency_ms()
    latency_display = f"{latency_ms:.1f} ms" if latency_ms is not None else "Pending Benchmark Run"

    st.markdown('<div class="qknee-eyebrow">Capability Summary</div>', unsafe_allow_html=True)
    card_col1, card_col2, card_col3 = st.columns(3)
    with card_col1:
        st.markdown(
            f"""
            <div class="qknee-card">
                <div class="qknee-card-title">High-Dimensional Feature Extraction</div>
                <div class="qknee-card-body">
                    ResNet18 convolutional backbone, ImageNet-pretrained and frozen, projecting each
                    slice into a {feature_dim}-dimensional embedding — the radiological ingestion
                    pipeline every downstream stage consumes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card_col2:
        st.markdown(
            f"""
            <div class="qknee-card">
                <div class="qknee-card-title">{n_qubits}-Qubit Variational Quantum Kernel</div>
                <div class="qknee-card-body">
                    Continuous-variable angle encoding into a {n_layers}-layer entangled circuit —
                    a {reduction_pct:.1f}% trainable-parameter reduction versus an equivalent
                    classical bottleneck. Measured per-sample latency: {latency_display}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with card_col3:
        st.markdown(
            """
            <div class="qknee-card">
                <div class="qknee-card-title">Spatial Explainability Engine</div>
                <div class="qknee-card-body">
                    Gradient-Weighted Class Activation Mapping (Grad-CAM) backpropagated from the
                    predicted risk score itself, for quantitative lesion localization and
                    per-qubit attribution breakdown at inference time.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    cta_col1, cta_col2 = st.columns(2)
    with cta_col1:
        if st.button("Launch Diagnostic Workstation", type="primary", use_container_width=True):
            st.session_state[VIEW_STATE_KEY] = VIEW_DIAGNOSTIC
            st.rerun()
    with cta_col2:
        if st.button("Review Quantitative Benchmark Data", use_container_width=True):
            st.session_state[VIEW_STATE_KEY] = VIEW_BENCHMARK
            st.rerun()


# --------------------------------------------------------------------------- #
# 2. Product & architecture showcase — interactive 3-step pipeline explainer
# --------------------------------------------------------------------------- #

def render_pipeline_explainer() -> None:
    """Three-stage formal walkthrough of the Radiological Ingestion
    Pipeline -> Variational Quantum Kernel -> Attribution Breakdown &
    Report chain, with the offline-rendered circuit diagram and
    clinical-case-walkthrough figures (`scripts/generate_deck_assets.py`)
    embedded inline."""
    st.markdown("## System Architecture")
    st.caption(
        "Three stages, one forward pass: a classical vision backbone compresses each MRI slice, a "
        f"{_config.quantum.n_qubits}-qubit variational circuit scores tear risk, and a "
        "Grad-CAM/Pauli-Z attribution breakdown plus an auto-generated PDF report explain the result."
    )

    n_qubits = _config.quantum.n_qubits
    n_layers = _config.quantum.n_layers
    feature_dim = _config.resnet.feature_dim

    step1_tab, step2_tab, step3_tab = st.tabs([
        "Stage 01 — Radiological Ingestion Pipeline",
        "Stage 02 — Variational Quantum Kernel",
        "Stage 03 — Attribution Breakdown & Report",
    ])

    with step1_tab:
        text_col, fig_col = st.columns([1.1, 1])
        with text_col:
            st.markdown(
                f"""
**ResNet18 feature extraction.** Each MRI slice (or an aggregated
multi-slice volume) is passed through a frozen, ImageNet-pretrained
ResNet18 backbone, producing a dense **{feature_dim}-dimensional** feature
vector — the same radiological embedding every downstream stage
consumes (`qknee.models.resnet_extractor.ResNet18FeatureExtractor`).

- Frozen convolutional backbone — no retraining required
- {feature_dim}-D embedding per slice
- Multi-slice volumes aggregate via mean / attention / top-k pooling
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
                st.warning("Circuit diagram not yet generated. Run `python scripts/generate_deck_assets.py`.")

    with step3_tab:
        st.markdown(
            """
**Quantitative lesion localization + automated radiology report.** Grad-CAM
performs quantitative lesion localization, highlighting which spatial
regions of the slice drove the ResNet18 embedding; the VQC's per-qubit
Pauli-Z measurements supply a formal attribution breakdown of the quantum
circuit's own contribution; and `qknee.xai.report_generator` compiles the
slice, the Grad-CAM overlay, and the full quantum/clinical attribution
breakdown into a structured, downloadable diagnostic PDF report.
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
# 3. Interactive live sample showcase — 1-click preview, no login required
# --------------------------------------------------------------------------- #

_PLANE_ABBREVIATIONS = {"axial": "AX", "coronal": "COR", "sagittal": "SAG"}

_INDICATION_BY_CATEGORY = {
    "normal": "Routine Surveillance MRI — No Acute Findings",
    "acl tear": "Suspected ACL Tear",
    "meniscal tear": "Suspected Meniscal Tear",
    "mcl sprain": "Suspected MCL Sprain",
}


def _study_uid(case: Dict) -> str:
    """Formal DICOM-style study identifier, e.g. `MRNet-SAG-0428`,
    derived deterministically from the case's plane and numeric suffix of
    `case_id` so it stays stable across reruns."""
    plane_abbrev = _PLANE_ABBREVIATIONS.get(str(case.get("plane", "")).lower(), "UNK")
    digits = "".join(ch for ch in str(case.get("case_id", "0")) if ch.isdigit()) or "0000"
    return f"MRNet-{plane_abbrev}-{digits[-4:].zfill(4)}"


def _clinical_indication(case: Dict) -> str:
    category = str(case.get("ground_truth_category", "")).strip().lower()
    return _INDICATION_BY_CATEGORY.get(category, f"Suspected {case.get('ground_truth_category', 'Finding')}")


def render_case_study_table(cases: List[Dict]) -> None:
    """Structured, formal case-study table (Study UID / Acquisition Plane
    / Clinical Indication / Reference Classification) — the "Interactive
    Diagnostic Preview" summary a radiology-facing visitor expects before
    drilling into any one case's overlay below."""
    import pandas as pd

    rows = [
        {
            "Study UID": _study_uid(case),
            "Acquisition Plane": str(case.get("plane", "?")).capitalize(),
            "Clinical Indication": _clinical_indication(case),
            "Reference Classification": case.get("ground_truth_category", "Unknown"),
            "Model Risk Tier": case.get("risk_tier", "N/A"),
        }
        for case in cases
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_live_sample_showcase() -> None:
    """Interactive Diagnostic Preview: a structured case-study table
    followed by 3 pre-scored knee MRI reference cases (one normal
    surveillance study, one ACL tear, one meniscal tear), previewable
    on demand straight from `precomputed_cache.json` — no login, no
    upload, no live inference."""
    st.markdown("## Interactive Diagnostic Preview")
    st.caption(
        "Reference case studies drawn from the precomputed inference cache — zero login, zero "
        "upload, zero inference latency."
    )

    all_cases = _load_precomputed_cases()
    by_id = {case["case_id"]: case for case in all_cases}
    showcase_cases = [by_id[case_id] for case_id in SHOWCASE_CASE_IDS if case_id in by_id]

    if not showcase_cases:
        st.info(
            "No precomputed sample cases found yet. Run `python scripts/generate_demo_cache.py` "
            "to build the showcase cache."
        )
        return

    render_case_study_table(showcase_cases)
    st.write("")

    columns = st.columns(len(showcase_cases))
    for column, case in zip(columns, showcase_cases):
        with column:
            with st.container(border=True):
                st.markdown(f"**{_study_uid(case)}**")
                st.caption(_clinical_indication(case))

                preview_key = f"_qknee_showcase_preview_{case['case_id']}"
                if st.button(
                    "View Case Detail", key=f"qknee_showcase_btn_{case['case_id']}", use_container_width=True,
                ):
                    st.session_state[preview_key] = True

                if st.session_state.get(preview_key):
                    overlay_bgr = _decode_case_overlay(case)
                    if overlay_bgr is not None:
                        st.image(
                            overlay_bgr[:, :, ::-1],  # BGR -> RGB for st.image
                            caption="Quantitative Lesion Localization (Grad-CAM Overlay)",
                            use_container_width=True,
                        )
                    else:
                        st.warning("Attribution overlay unavailable for this case.")

                    risk_score = float(case.get("risk_score", 0.0))
                    st.metric(
                        "Composite Tear Risk",
                        f"{risk_score * 100:.1f}%",
                        delta=case.get("risk_tier", "N/A"),
                        delta_color="inverse" if case.get("risk_tier") == "LOW" else "off",
                    )
                    st.caption(case.get("classification_label", ""))

    st.caption(
        f"Every study above was produced by the real Q-Knee pipeline "
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
