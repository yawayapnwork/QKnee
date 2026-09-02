"""
Q-Knee diagnostic web app (Streamlit) — a lighter, button-triggered
alternative to the always-on clinical dashboard in app.py, aimed at
interactive single-scan review: upload a scan, adjust how you're viewing
it, then explicitly trigger the quantum analysis.

Differences from app.py (which auto-runs inference on every slice change
and shows dual ACL/meniscus scores):
    - Upload supports .png/.jpg/.jpeg/.npy (no DICOM here).
    - A contrast slider lets you re-window the displayed slice without
      re-running inference.
    - Analysis only runs when you click "Run Q-Knee Analysis" — the
      (potentially expensive) ResNet18 + quantum-circuit forward pass
      doesn't fire on every slider drag.
    - A single overall "Normal" / "Abnormality Detected" verdict with a
      radial risk gauge and a highlighted quantum-circuit latency figure,
      rather than per-condition risk bars.

Falls back to a deterministic mock pipeline if the trained PipelineRunner
backend (pca_scaler.pkl) isn't available, so the UI is fully demoable
without a live backend.

Run with:
    streamlit run qknee_frontend.py
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set before matplotlib's own import (not after): a container's default
# config dir is often read-only or lives on slow/non-tmpfs storage, and
# matplotlib builds its font cache there on first import — on a cold boot
# that can stall for tens of seconds. Pointing it at a tmpfs-backed `/tmp`
# subdirectory up front makes that one-time build (and every import after
# the first) fast. `setdefault` so an operator-supplied override wins.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger
from qknee.ui import theme

_config = load_config()
logger = get_logger(__name__)
RISK_THRESHOLD = _config.api.tear_risk_threshold
N_QUBITS = _config.quantum.n_qubits

# All demo/deck assets are resolved relative to the repo root (this file's
# grandparent directory: qknee/ui/analysis_app.py -> qknee/ -> repo root),
# never the process's current working directory — `streamlit run` doesn't
# guarantee cwd is the repo root, and a bare relative
# `Path("qknee/artifacts/...")` silently resolves to nothing (no error,
# just `.exists() == False`) when it isn't.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARTIFACTS_DIR = _REPO_ROOT / "qknee" / "artifacts"
DEMO_CACHE_DIR = _ARTIFACTS_DIR / "demo_cache"
DECK_FIGURES_DIR = _ARTIFACTS_DIR / "deck_figures"

# The PRD's Plan B latency-risk mitigation cache (`scripts/generate_demo_cache.py`)
# — 10 cases' full pipeline outputs serialized to one JSON file, each Grad-CAM
# heatmap embedded as base64 so nothing needs a second filesystem read. Served
# directly by the "Judge Mode" sidebar toggle (`render_fast_path_sidebar`) so a
# live demo never blocks on a cold model/QNode load or slow CPU inference.
PRECOMPUTED_CACHE_PATH = _ARTIFACTS_DIR / "precomputed_cache.json"

PLANES = ["Axial", "Coronal", "Sagittal"]

# Multi-slice Grad-CAM runs one full ResNet18 forward + backward pass per
# slice — a volume deep enough to matter (dozens to low-hundreds of
# slices) would otherwise block the UI for a long time on CPU. Volumes
# with more slices than this along the chosen plane are evenly
# subsampled down to this many before running Grad-CAM, so the operation
# stays interactive; `VolumetricAnalysisResult` reports which real volume
# indices were actually analyzed.
MAX_VOLUMETRIC_SLICES = 40


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #

@dataclass
class AnalysisResult:
    risk_score: float             # overall (worst-case) risk driving the single verdict badge/gauge
    prediction_label: str
    quantum_latency_ms: float
    total_latency_ms: float
    backend: str  # "live" or "mock"
    gradcam_overlay: Optional[np.ndarray] = None  # (H, W, 3) BGR uint8 pre-blended overlay, or None
    gradcam_heatmap: Optional[np.ndarray] = None  # (h, w) float32 in [0, 1], raw Grad-CAM — lets the
    # UI re-blend at any opacity live (see the opacity slider in `main()`) without re-running inference.
    pauli_z_expectations: Optional[np.ndarray] = None  # (n_qubits,) in [-1, 1] — the quantum circuit's
    # own raw per-qubit output, read before the classical Linear+Sigmoid readout collapses it to a
    # single risk probability. None if the VQC backbone doesn't expose one (see get_pauli_z_expectations).
    # Per-condition breakdown (the primary clinical triad — matches
    # qknee.models.vqc_multitarget.TRIAD_CONDITIONS) for the Quantum Decision
    # Metrics panel; None for a condition whose head isn't available for this
    # backend/case (e.g. an API-only or precomputed-cache result).
    acl_risk: Optional[float] = None
    mcl_risk: Optional[float] = None
    meniscus_risk: Optional[float] = None


@dataclass
class VolumetricAnalysisResult:
    """Multi-slice Grad-CAM over an entire anatomical plane's worth of
    slices (see `qknee.xai.gradcam.compute_volumetric_gradcam`) — powers
    the interactive slice explorer's real-time (zero-recompute) heatmap
    scrubbing and the top-3 high-salience-slice highlighting."""

    plane: str
    heatmaps: Dict[int, np.ndarray] = field(default_factory=dict)   # slice_index -> (h, w) heatmap in [0, 1]
    magnitudes: Dict[int, float] = field(default_factory=dict)      # slice_index -> raw salience magnitude
    top_k_indices: List[int] = field(default_factory=list)          # highest-magnitude slice indices, descending
    analyzed_indices: List[int] = field(default_factory=list)       # which volume indices were actually analyzed
    backend: str = "live"                                            # "live" or "mock"

    def overlay_for(
        self, slice_index: int, display_slice: np.ndarray, alpha: float, colormap: int,
    ) -> Optional[np.ndarray]:
        """Blends this slice's precomputed heatmap onto `display_slice` at
        any `alpha`/`colormap` — a resize + colormap + `cv2.addWeighted`,
        effectively free, so scrubbing the slice slider or dragging the
        alpha/colormap controls after this result has been computed never
        re-runs inference. Returns `None` if `slice_index` wasn't analyzed."""
        heatmap = self.heatmaps.get(slice_index)
        if heatmap is None:
            return None
        from qknee.xai.gradcam import overlay_heatmap

        return overlay_heatmap(heatmap, display_slice, alpha=alpha, colormap=colormap)


def get_pauli_z_expectations(vqc_model, quantum_angles: np.ndarray) -> Optional[np.ndarray]:
    """Raw per-qubit Pauli-Z expectation values in `[-1, 1]` — the 4-qubit
    circuit's own measurement output, read directly from `vqc_model`'s
    `quantum_layer` before the classical `Linear(n_qubits, 1)` + sigmoid
    readout collapses it into a single risk probability. This is what the
    Quantum State Attribution panel plots.

    Prefers `vqc_model.predict_fast()` (duck-typed via `hasattr`) when
    available: `run_live_analysis` below already called `runner.classify()`
    on this exact angle vector moments earlier, which — for a
    `VQCClassifier` — runs through `predict_fast` and populates its
    per-instance angle cache; calling `predict_fast` again here for the
    *same* angles is then a cache hit (sub-millisecond), rather than
    redundantly re-running the full slow `TorchLayer`/`default.qubit`/
    `backprop` circuit a second time just to read back the expectation
    values `classify()` already computed and discarded. Falls back to the
    original direct `quantum_layer(...)` call for any model without
    `predict_fast` (e.g. `DataReuploadingVQC`).

    Returns `None` (rather than guessing) if `vqc_model` doesn't expose a
    `.quantum_layer` attribute — e.g. a custom ansatz that doesn't follow
    `VQCClassifier`'s structure.
    """
    if hasattr(vqc_model, "predict_fast"):
        try:
            _, expvals = vqc_model.predict_fast(np.asarray(quantum_angles).reshape(-1))
            return expvals
        except Exception as exc:  # noqa: BLE001 - fall through to the direct quantum_layer call below
            logger.warning("predict_fast failed for Pauli-Z readout (%s); using quantum_layer directly.", exc)

    quantum_layer = getattr(vqc_model, "quantum_layer", None)
    if quantum_layer is None:
        return None

    import torch

    with torch.inference_mode():
        angles_tensor = torch.from_numpy(np.asarray(quantum_angles)).float()
        expvals = quantum_layer(angles_tensor)
    return expvals.detach().cpu().numpy().reshape(-1)


def _release_inference_memory() -> None:
    """Frees intermediate tensors/buffers left over from a heavy
    inference or Grad-CAM pass — a Streamlit Community Cloud container's
    free tier caps out around 1GB, and this app is meant to stay
    comfortably under a ~600MB resident footprint across a long-running
    session with many uploads, not just at cold start.

    `gc.collect()` unconditionally (a ResNet18 forward pass and Grad-CAM's
    backward pass can leave short-lived reference cycles that outlive the
    call that created them); `torch.cuda.empty_cache()` additionally when
    CUDA is actually present — this project's pinned `requirements.txt` is
    CPU-only torch by default, so on the typical deployment this is just
    the `gc.collect()`, with the CUDA branch as a no-op safety net for
    anyone running a GPU-enabled build locally.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Backend loading + inference (mock-fallback pattern shared with app.py)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False, max_entries=1)
def load_backend():
    """Loads a `PipelineRunner` (DataIngestion -> ResNet18 -> PCA -> VQC ->
    GradCAM) if a fitted PCA artifact is available; returns None otherwise
    so the UI can fall back to a deterministic mock. `max_entries=1`: no
    arguments, so there is only ever one possible entry — the ResNet18
    backbone + VQC this loads is exactly the kind of heavyweight resource
    the 1GB Streamlit Cloud ceiling means never to accidentally duplicate."""
    if not _config.paths.pca_artifact.exists():
        return None

    try:
        from qknee.models.pipeline import PipelineRunner

        return PipelineRunner(config=_config)
    except Exception as exc:  # noqa: BLE001
        st.session_state["_backend_error"] = str(exc)
        return None


@st.cache_resource(show_spinner=False, max_entries=1)
def load_condition_models() -> Optional[Dict[str, object]]:
    """Three independent quantum heads for the primary clinical triad
    (ACL / MCL / Meniscus — matches `qknee.models.vqc_multitarget.
    TRIAD_CONDITIONS`), used by the Quantum Decision Metrics panel's
    per-condition risk breakdown (`run_live_analysis` feeds the same
    quantum-angle vector `runner.classify()`s each on, so all three scores
    plus the overall verdict come from one ResNet18/PCA forward pass).

    ACL and Meniscus each load their own trained checkpoint from
    `config.yaml`'s `paths.acl_checkpoint`/`paths.meniscus_checkpoint` when
    available; MCL has no dedicated checkpoint path (yet) and always uses
    seeded-random weights. Returns `None` if torch/pennylane aren't
    importable, so the caller can render "N/A" badges instead."""
    try:
        import torch

        from qknee.models.pipeline import PipelineValidationError, load_vqc_weights
        from qknee.models.vqc import VQCClassifier
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build condition-specific VQC heads: %s", exc)
        return None

    torch.manual_seed(42)
    acl_model = VQCClassifier()
    if _config.paths.acl_checkpoint.exists():
        try:
            load_vqc_weights(acl_model, _config.paths.acl_checkpoint)
        except PipelineValidationError as exc:
            logger.warning("Failed to load ACL checkpoint (%s); using random weights: %s",
                            _config.paths.acl_checkpoint, exc)
    acl_model.eval()

    torch.manual_seed(21)
    mcl_model = VQCClassifier()
    mcl_model.eval()

    torch.manual_seed(7)
    meniscus_model = VQCClassifier()
    if _config.paths.meniscus_checkpoint.exists():
        try:
            load_vqc_weights(meniscus_model, _config.paths.meniscus_checkpoint)
        except PipelineValidationError as exc:
            logger.warning("Failed to load meniscus checkpoint (%s); using random weights: %s",
                            _config.paths.meniscus_checkpoint, exc)
    meniscus_model.eval()

    return {"ACL": acl_model, "MCL": mcl_model, "Meniscus": meniscus_model}


@st.cache_resource(show_spinner=False, max_entries=1)
def load_precomputed_cache() -> Optional[dict]:
    """Judge Fast-Path source: `qknee/artifacts/precomputed_cache.json`'s
    pre-scored cases (Grad-CAM overlay embedded as base64), so the "Enable
    NISQ Simulation Fast-Path" toggle can serve a result with zero model
    load and zero QNode execution — insurance against a live judging
    session timing out on a cold backend or slow CPU inference. Returns
    `None` (logged) if the cache hasn't been built yet
    (`python scripts/generate_demo_cache.py`) or fails to parse."""
    import json

    if not PRECOMPUTED_CACHE_PATH.exists():
        logger.info("No precomputed cache found at %s; Judge Fast-Path disabled.", PRECOMPUTED_CACHE_PATH)
        return None
    try:
        return json.loads(PRECOMPUTED_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read precomputed cache at %s: %s", PRECOMPUTED_CACHE_PATH, exc)
        return None


def render_fast_path_sidebar() -> Tuple[bool, Optional[dict]]:
    """Renders the Clinical Action Bar's 'Toggle NISQ Acceleration Cache'
    control — bypasses upload/live/mock inference entirely and replays
    one of `precomputed_cache.json`'s pre-scored cases straight from
    process memory, for a low-latency demonstration mode. Returns
    `(use_fast_path, selected_case)`."""
    st.sidebar.markdown("### Clinical Action Bar")
    st.sidebar.caption("NISQ Acceleration Cache")

    cache = load_precomputed_cache()
    cases: List[dict] = (cache or {}).get("cases", [])

    use_fast_path = st.sidebar.toggle(
        "Toggle NISQ Acceleration Cache",
        value=False,
        disabled=not cases,
        help=f"Instantly replays one of {len(cases)} precomputed case(s) from "
             f"`{PRECOMPUTED_CACHE_PATH}` — zero model load, zero QNode "
             "execution — a low-latency demonstration mode for time-constrained review.",
    )

    if not cases:
        st.sidebar.caption(
            f"No precomputed cache found at `{PRECOMPUTED_CACHE_PATH}`. Run "
            "`python scripts/generate_demo_cache.py` to build one."
        )
        return False, None
    if not use_fast_path:
        st.sidebar.caption(f"{len(cases)} accelerated case(s) available.")
        return False, None

    case_labels = [f"{case['case_id']} ({case.get('plane', '?')})" for case in cases]
    selected_label = st.sidebar.selectbox("Accelerated Case", case_labels)
    selected_case = cases[case_labels.index(selected_label)]
    st.sidebar.success(f"Serving '{selected_case['case_id']}' from the acceleration cache — 0 ms inference.")
    return True, selected_case


def build_fast_path_result(case: dict) -> Tuple[Optional[np.ndarray], AnalysisResult]:
    """Decodes one `precomputed_cache.json` case's base64-embedded Grad-CAM
    overlay (already resized/colormapped/blended at generation time) —
    genuinely zero-latency: no live ResNet18/PCA/VQC forward pass, and no
    second filesystem read (the base64 payload is preferred; a
    `heatmap_file` fallback is used only if the embedded payload is
    missing)."""
    import base64

    import cv2

    overlay: Optional[np.ndarray] = None
    heatmap_b64 = case.get("heatmap_base64")
    if heatmap_b64:
        try:
            png_bytes = base64.b64decode(heatmap_b64)
            overlay = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001 - fall back to the on-disk copy below
            logger.warning("Failed to decode base64 heatmap for case %s: %s", case.get("case_id"), exc)

    if overlay is None and case.get("heatmap_file"):
        heatmap_path = _ARTIFACTS_DIR / case["heatmap_file"]
        if heatmap_path.exists():
            overlay = cv2.imread(str(heatmap_path), cv2.IMREAD_COLOR)

    risk_score = float(case["risk_score"])
    label = "Abnormality Detected" if risk_score >= RISK_THRESHOLD else "Normal"
    raw_pauli_z = case.get("pauli_z_expectations")
    pauli_z_expectations = np.asarray(raw_pauli_z, dtype=np.float32) if raw_pauli_z else None

    result = AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=float(case.get("quantum_latency_ms", 0.0)),
        total_latency_ms=0.0,  # served straight from cache — no recomputation happened this request
        backend=f"cached-fastpath/{case.get('backend', 'unknown')}",
        gradcam_overlay=overlay,
        gradcam_heatmap=None,  # only the pre-blended overlay is cached, not the raw per-pixel heatmap
        pauli_z_expectations=pauli_z_expectations,
        # precomputed_cache.json scores one unified risk, not a per-condition
        # ACL/MCL/Meniscus breakdown — the panel renders "N/A" for MCL/Meniscus.
        acl_risk=risk_score,
        mcl_risk=None,
        meniscus_risk=None,
    )
    return overlay, result


def render_fast_path_view(case: dict) -> None:
    """Renders a lightweight, fully self-contained view of one NISQ
    Acceleration Cache case — bypasses upload/plane/slice controls
    entirely, since the cached case carries its own precomputed
    image/scores."""
    st.info(
        f"NISQ Acceleration Cache active — serving precomputed case **{case['case_id']}** "
        "from cache (0 ms inference)."
    )
    overlay, result = build_fast_path_result(case)

    image_col, gauge_col, attrib_col = st.columns([1, 1, 1])

    with image_col:
        st.markdown(
            f"#### {theme.icon('crosshair', size=17)} Quantitative Lesion Localization "
            f"— {case.get('plane', '?').title()} Plane",
            unsafe_allow_html=True,
        )
        if overlay is not None:
            snippet = case.get("clinical_text_snippet")
            caption = snippet.splitlines()[0] if snippet else None
            st.image(overlay, channels="BGR", width="stretch", caption=caption)
        else:
            st.warning("No cached attribution overlay available for this case.")

    with gauge_col:
        render_prediction_badge(result)
        st.pyplot(render_risk_gauge(result.risk_score), width="stretch")
        st.metric("Quantum Circuit Latency (Cached)", format_latency_ms(result.quantum_latency_ms))
        st.caption(f"Backend: **{result.backend}**")

    with attrib_col:
        st.markdown("##### Quantitative Clinical Triage Panel")
        badge_cols = st.columns(3)
        with badge_cols[0]:
            render_condition_risk_badge("ACL Tear", result.acl_risk)
        with badge_cols[1]:
            render_condition_risk_badge("MCL Sprain", result.mcl_risk)
        with badge_cols[2]:
            render_condition_risk_badge("Medial Meniscus Tear", result.meniscus_risk)
        fig = render_quantum_attribution_panel(result.pauli_z_expectations)
        if fig is not None:
            st.pyplot(fig, width="stretch")
            st.caption("Per-qubit Pauli-Z expectation ⟨Z⟩, precomputed offline for this case.")


def render_report_download(display_slice: np.ndarray, result: AnalysisResult) -> None:
    """Clinical Action Bar item: "Generate Formal Diagnostic Report (PDF)"
    — compiles the current slice, Grad-CAM overlay, and ACL/MCL/meniscus
    risk breakdown into a one-page radiology-style PDF via
    `qknee.xai.report_generator`, offered as a direct download. A failed
    generation degrades to a warning rather than crashing the workstation."""
    from qknee.xai.gradcam import overlay_heatmap
    from qknee.xai.report_generator import generate_radiology_report

    gradcam_overlay = result.gradcam_overlay
    if gradcam_overlay is None and result.gradcam_heatmap is not None:
        gradcam_overlay = overlay_heatmap(result.gradcam_heatmap, display_slice)

    try:
        pdf_bytes = generate_radiology_report(
            output_path=None,
            mri_slice=display_slice,
            gradcam_overlay=gradcam_overlay,
            prediction_results={
                "acl_risk": result.acl_risk,
                "mcl_risk": result.mcl_risk,
                "meniscus_risk": result.meniscus_risk,
                "pauli_z_expectations": (
                    result.pauli_z_expectations.tolist() if result.pauli_z_expectations is not None else None
                ),
                "quantum_latency_ms": result.quantum_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "backend": result.backend,
            },
            metadata={
                "modality": "MRI Knee",
                "clinical_indication": "Q-Knee diagnostic workstation session",
                "scan_date": datetime.now().strftime("%Y-%m-%d"),
            },
        )
    except Exception as exc:  # noqa: BLE001 - a failed report shouldn't crash the workstation
        logger.warning("PDF report generation failed: %s", exc)
        st.warning("Unable to generate the formal diagnostic report for this study.")
        return
    finally:
        # reportlab's Canvas, the PIL images it wraps, and the in-memory
        # PNG/PDF byte buffers `generate_radiology_report` builds are all
        # done being used past this point — release them promptly rather
        # than waiting on the next GC cycle, since this runs right after
        # the single heaviest call on the hot path (live inference).
        _release_inference_memory()

    st.download_button(
        label="Export Standard Clinical Report (PDF)",
        data=pdf_bytes,
        file_name=f"qknee_diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        width="stretch",
        type="primary",
    )


def render_deck_figures_expander() -> None:
    """Surfaces the offline-generated deck figures (`scripts/generate_deck_assets.py`)
    — e.g. the circuit diagram — inline for reference, when available."""
    figure_path = DECK_FIGURES_DIR / "circuit_diagram.png"
    if not figure_path.exists():
        return
    with st.expander("Quantum Circuit Diagram (Reference)"):
        st.image(str(figure_path), width="stretch")


def _mock_gradcam_heatmap(slice_2d: np.ndarray) -> np.ndarray:
    """Cheap, torch-free stand-in *raw* Grad-CAM heatmap for mock mode (a
    soft radial gradient), so the heatmap panel — including its live
    opacity slider — is exercised in the UI even without a live backend.
    Blended into a displayable overlay via `qknee.xai.gradcam.overlay_heatmap`,
    the same function the live backend uses."""
    display = normalize_uint8(slice_2d)
    height, width = display.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2, width / 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return np.clip(1 - radius / radius.max(), 0, 1).astype(np.float32)


def run_live_analysis(runner, slice_2d: np.ndarray) -> AnalysisResult:
    """Delegates to `PipelineRunner`'s stage methods so this UI shares the
    exact same validated ingestion/ResNet18/PCA/VQC path as the API, and
    generates a Grad-CAM overlay backpropagated from the predicted risk
    score itself (`PipelineRunner.explain()`) — not embedding energy — so
    the heatmap explains that prediction."""
    from qknee.xai.gradcam import overlay_heatmap

    # perf_counter_ns() rather than perf_counter(): the quantum stage alone
    # now regularly lands under 1ms on a `VQCClassifier.predict_fast` cache
    # hit (see qknee.models.vqc) — float-seconds precision starts losing
    # real signal at that scale, the nanosecond counter doesn't.
    t0_ns = time.perf_counter_ns()
    batch = runner.ingest(slice_2d)
    features = runner.extract_resnet_features(batch)
    quantum_angles = runner.reduce_to_quantum_angles(features)
    t1_ns = time.perf_counter_ns()
    risk_score = runner.classify(quantum_angles)
    t2_ns = time.perf_counter_ns()

    gradcam_overlay: Optional[np.ndarray] = None
    gradcam_heatmap: Optional[np.ndarray] = None
    try:
        gradcam_heatmap = runner.explain(batch[:, 0])
        gradcam_overlay = overlay_heatmap(gradcam_heatmap, slice_2d)
    except Exception as exc:  # noqa: BLE001 - a failed heatmap shouldn't hide the risk score
        logger.warning("Grad-CAM generation failed; showing risk score without an overlay: %s", exc)

    quantum_latency_ms = (t2_ns - t1_ns) / 1e6
    total_latency_ms = (t2_ns - t0_ns) / 1e6
    label = "Abnormality Detected" if risk_score >= RISK_THRESHOLD else "Normal"
    pauli_z_expectations = get_pauli_z_expectations(runner.vqc, quantum_angles)

    # Per-condition breakdown for the Quantum Decision Metrics panel: reuse
    # the same quantum_angles vector (no extra ResNet18/PCA work) against
    # each of the three condition-specific heads.
    acl_risk = mcl_risk = meniscus_risk = None
    condition_models = load_condition_models()
    if condition_models is not None:
        acl_risk = runner.classify(quantum_angles, vqc=condition_models["ACL"])
        mcl_risk = runner.classify(quantum_angles, vqc=condition_models["MCL"])
        meniscus_risk = runner.classify(quantum_angles, vqc=condition_models["Meniscus"])

    result = AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=quantum_latency_ms,
        total_latency_ms=total_latency_ms,
        backend="live",
        gradcam_overlay=gradcam_overlay,
        gradcam_heatmap=gradcam_heatmap,
        pauli_z_expectations=pauli_z_expectations,
        acl_risk=acl_risk,
        mcl_risk=mcl_risk,
        meniscus_risk=meniscus_risk,
    )

    # `batch`/`features`/`quantum_angles` and Grad-CAM's backward-pass
    # activation graph are done being used by this point — every value
    # this function returns has already been pulled out into plain
    # numpy/float `AnalysisResult` fields above.
    del batch, features, quantum_angles
    _release_inference_memory()

    return result


def run_mock_analysis(slice_2d: np.ndarray) -> AnalysisResult:
    """Deterministic (seeded from slice content) mock, used when no trained
    backend is available."""
    from qknee.xai.gradcam import overlay_heatmap

    digest = hashlib.sha256(slice_2d.tobytes()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)

    quantum_latency_ms = float(rng.uniform(4, 12))
    total_latency_ms = quantum_latency_ms + float(rng.uniform(20, 35))
    risk_score = float(rng.uniform(0.05, 0.95))
    label = "Abnormality Detected" if risk_score >= RISK_THRESHOLD else "Normal"
    heatmap = _mock_gradcam_heatmap(slice_2d)
    pauli_z_expectations = rng.uniform(-1.0, 1.0, size=N_QUBITS).astype(np.float32)

    return AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=quantum_latency_ms,
        total_latency_ms=total_latency_ms,
        backend="mock",
        gradcam_overlay=overlay_heatmap(heatmap, slice_2d),
        gradcam_heatmap=heatmap,
        pauli_z_expectations=pauli_z_expectations,
        acl_risk=float(rng.uniform(0.05, 0.95)),
        mcl_risk=float(rng.uniform(0.05, 0.95)),
        meniscus_risk=float(rng.uniform(0.05, 0.95)),
    )


def run_analysis(slice_2d: np.ndarray) -> AnalysisResult:
    runner = load_backend()
    if runner is not None:
        return run_live_analysis(runner, slice_2d)
    return run_mock_analysis(slice_2d)


# --------------------------------------------------------------------------- #
# Multi-slice (volumetric) Grad-CAM
# --------------------------------------------------------------------------- #

def _subsampled_indices(num_slices: int, max_slices: int) -> List[int]:
    """Evenly-spaced indices covering `[0, num_slices)`, capped at
    `max_slices` — keeps `run_volumetric_gradcam_live` interactive on a
    deep volume by analyzing a representative subset rather than every
    slice."""
    if num_slices <= max_slices:
        return list(range(num_slices))
    return sorted(set(np.linspace(0, num_slices - 1, num=max_slices, dtype=int).tolist()))


def _risk_target_fn_for(runner):
    """Builds a Grad-CAM `target_fn` that continues the forward pass from
    a ResNet embedding through `runner`'s dimensionality-reduction stage
    (PCA or the quantum autoencoder) and its VQC to the scalar predicted
    risk probability — mirrors `PipelineRunner`'s own internal
    `_risk_target_fn` (used by `PipelineRunner.explain()`) so a
    multi-slice heatmap explains the same risk score the single-slice
    Grad-CAM path does, just built here from `runner`'s public
    `pca_layer`/`quantum_autoencoder`/`vqc` attributes so this UI module
    doesn't need to reach into PipelineRunner's private API."""
    def risk_target(resnet_output):
        if runner.encoder_type == "pca":
            angles = runner.pca_layer(resnet_output)
        else:
            angles = runner.quantum_autoencoder.compress(resnet_output)
        risk = runner.vqc(angles)
        return risk.squeeze()

    return risk_target


def run_volumetric_gradcam_live(
    runner, volume: np.ndarray, plane: str, max_slices: int = MAX_VOLUMETRIC_SLICES, top_k: int = 3,
) -> VolumetricAnalysisResult:
    """Runs real Grad-CAM independently across every slice (or an evenly
    subsampled subset, if the volume is deep — see `_subsampled_indices`)
    of `volume` along `plane`, via `qknee.xai.gradcam.compute_volumetric_gradcam`.

    Each slice is explained against the same risk-score target
    (`_risk_target_fn_for`), so the resulting per-slice heatmaps and
    salience ranking are directly comparable to each other and to the
    single-slice Grad-CAM panel driven by `run_live_analysis`.
    """
    from qknee.data.ingestion import MultiPlaneViewSelector
    from qknee.xai.gradcam import compute_volumetric_gradcam

    selector = MultiPlaneViewSelector(volume)
    num_slices = selector.num_slices(plane.lower())
    indices = _subsampled_indices(num_slices, max_slices)

    slice_tensors = []
    for index in indices:
        raw_slice = selector.get_slice(plane.lower(), index)
        batch = runner.ingest(raw_slice)   # (1, S, 3, 224, 224)
        slice_tensors.append(batch[:, 0])  # (1, 3, 224, 224) — the ingested representative slice

    result = compute_volumetric_gradcam(
        model=runner.feature_extractor,
        target_layer=runner.gradcam_target_layer,
        slice_tensors=slice_tensors,
        slice_indices=indices,
        target_fn=_risk_target_fn_for(runner),
        top_k=top_k,
    )

    volumetric_result = VolumetricAnalysisResult(
        plane=plane,
        heatmaps={s.slice_index: s.heatmap for s in result.saliencies},
        magnitudes={s.slice_index: s.magnitude for s in result.saliencies},
        top_k_indices=result.top_k_indices,
        analyzed_indices=indices,
        backend="live",
    )

    # This is the single heaviest call in the app: up to `max_slices`
    # (default 40) full ResNet18 forward + Grad-CAM backward passes, each
    # holding its own `(1, 3, 224, 224)` ingested tensor (`slice_tensors`)
    # and activation graph. Every value needed downstream has already been
    # pulled into `volumetric_result`'s plain numpy dict fields above, so
    # release the rest before returning rather than leaving up to 40
    # tensors' worth of graph/activations for the next GC cycle to find.
    del slice_tensors, result
    _release_inference_memory()

    return volumetric_result


def run_volumetric_gradcam_mock(
    volume: np.ndarray, plane: str, max_slices: int = MAX_VOLUMETRIC_SLICES, top_k: int = 3,
) -> VolumetricAnalysisResult:
    """Torch-free stand-in for `run_volumetric_gradcam_live`, used when no
    trained backend is available — a per-slice radial-gradient heatmap
    (matching `_mock_gradcam_heatmap`'s single-slice mock) plus a
    deterministic, content-seeded magnitude per slice, so the multi-slice
    explorer (including top-3 salience highlighting) is fully exercisable
    without a live backend."""
    from qknee.data.ingestion import MultiPlaneViewSelector

    selector = MultiPlaneViewSelector(volume)
    num_slices = selector.num_slices(plane.lower())
    indices = _subsampled_indices(num_slices, max_slices)

    heatmaps: Dict[int, np.ndarray] = {}
    magnitudes: Dict[int, float] = {}
    for index in indices:
        raw_slice = selector.get_slice(plane.lower(), index)
        heatmaps[index] = _mock_gradcam_heatmap(raw_slice)
        digest = hashlib.sha256(raw_slice.tobytes()).digest()
        seed = int.from_bytes(digest[:4], "big")
        magnitudes[index] = float(np.random.default_rng(seed).uniform(0, 1000))

    top_k_indices = sorted(magnitudes, key=lambda i: magnitudes[i], reverse=True)[: max(top_k, 0)]

    return VolumetricAnalysisResult(
        plane=plane,
        heatmaps=heatmaps,
        magnitudes=magnitudes,
        top_k_indices=top_k_indices,
        analyzed_indices=indices,
        backend="mock",
    )


def run_volumetric_gradcam(volume: np.ndarray, plane: str) -> VolumetricAnalysisResult:
    runner = load_backend()
    if runner is not None:
        return run_volumetric_gradcam_live(runner, volume, plane)
    return run_volumetric_gradcam_mock(volume, plane)


# --------------------------------------------------------------------------- #
# Image loading + display adjustment
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Decoding uploaded scan...", max_entries=10, ttl=3600)
def _decode_scan_cached(file_payloads: Tuple[Tuple[str, bytes], ...]) -> np.ndarray:
    """Pure DICOM-series/.png/.jpg/.npy decode — the actual expensive step
    `load_scan` wraps. Keyed on `(filename, bytes)` tuples (real file
    content, hashable/comparable by `st.cache_data`) rather than Streamlit
    `UploadedFile` objects directly, and deliberately free of any `st.*`
    calls: Streamlit only replays a cached function's *return value* on a
    cache hit, not side effects like `st.error` performed inside it, so
    those must live in the uncached `load_scan` wrapper below instead —
    same split `qknee.ui.dashboard._decode_volume_cached`/`load_volume`
    uses for the same reason.

    `max_entries=10, ttl=3600`: bounds how many distinct decoded scans this
    session holds onto at once, and how long a stale one lingers, so
    repeated uploads across a long session don't grow this cache
    unbounded — see this module's 1GB-Streamlit-Cloud-ceiling budget.
    """
    import io

    from qknee.data.ingestion import DataIngestion

    if len(file_payloads) > 1:
        sources = []
        for name, content in file_payloads:
            buffer = io.BytesIO(content)
            buffer.name = name
            sources.append(buffer)
        return DataIngestion().load_volume_array(sources)

    name, content = file_payloads[0]
    buffer = io.BytesIO(content)
    buffer.name = name
    return DataIngestion().load_volume_array(buffer)


def load_scan(uploaded_files: List) -> Optional[np.ndarray]:
    """Loads one or more drag-and-dropped uploads into a `(D, H, W)`
    volume, via `qknee.data.ingestion.DataIngestion.load_volume_array`
    (through the cached `_decode_scan_cached` above) — the same
    DICOM-series/.npy/single-DICOM loading path `qknee.ui.dashboard` uses,
    so a multi-file `.dcm` series (one file per slice) stacks into a real
    tri-planar volume here too, not just a flat single image.

    Args:
        uploaded_files: One or more Streamlit `UploadedFile`s (`.png`,
            `.jpg`, `.jpeg`, `.npy`, or `.dcm`/`.dicom` — multiple `.dcm`
            files are treated as one series, sorted and stacked by
            `InstanceNumber`/`SliceLocation`).

    Returns `None` (and shows a Streamlit error) if the upload can't be parsed.
    """
    from qknee.data.ingestion import IngestionError

    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    if not files:
        return None

    display_name = f"{len(files)}-file DICOM series" if len(files) > 1 else files[0].name
    file_payloads = tuple((f.name, f.getvalue()) for f in files)

    try:
        array = _decode_scan_cached(file_payloads)
    except IngestionError as exc:
        st.error(f"Failed to read '{display_name}': {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read '{display_name}': {exc}")
        return None

    array = np.asarray(array)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    elif array.ndim != 3:
        st.error(f"Expected a 2D slice or 3D volume, got array of shape {array.shape}.")
        return None

    return array


def format_latency_ms(latency_ms: float) -> str:
    """Formats a millisecond latency for display, with enough precision to
    actually show a sub-millisecond `VQCClassifier.predict_fast` cache-hit
    value (e.g. "0.009 ms") rather than rounding it down to "0.0 ms" — one
    decimal place is fine above 1ms, but below that the fast path's whole
    point (near-zero latency) would otherwise be invisible in the UI."""
    if latency_ms < 1.0:
        return f"{latency_ms:.3f} ms"
    return f"{latency_ms:.1f} ms"


def normalize_uint8(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = slice_2d.astype(np.float32)
    min_val, max_val = float(slice_2d.min()), float(slice_2d.max())
    if max_val > min_val:
        slice_2d = (slice_2d - min_val) / (max_val - min_val)
    else:
        slice_2d = np.zeros_like(slice_2d)
    return (slice_2d * 255).astype(np.uint8)


def apply_contrast(slice_uint8: np.ndarray, contrast: float) -> np.ndarray:
    """Re-windows an 8-bit slice around its midpoint by `contrast`
    (1.0 = unchanged, >1 = higher contrast, <1 = flatter)."""
    adjusted = (slice_uint8.astype(np.float32) - 127.5) * contrast + 127.5
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Visual indicators
# --------------------------------------------------------------------------- #

def render_risk_gauge(risk_score: float) -> plt.Figure:
    """Renders a semicircular risk-score gauge (green/amber/red zones) with
    a needle pointing at the current score, via matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 2.4), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(theme.CARD_SURFACE)
    ax.set_facecolor(theme.CARD_SURFACE)

    # Sage-to-forest-green severity zones (not a rainbow scale) — low risk
    # is pale sage, high risk is muted terracotta, matching the badge palette.
    zones = [(0.0, 0.33, theme.SAGE_GREEN), (0.33, 0.66, theme.RISK_MODERATE), (0.66, 1.0, theme.RISK_HIGH)]
    for start, end, color in zones:
        ax.barh(
            1,
            (end - start) * np.pi,
            left=start * np.pi,
            height=0.4,
            color=color,
            edgecolor="none",
        )

    needle_angle = risk_score * np.pi
    ax.plot([needle_angle, needle_angle], [0, 1.15], color=theme.FOREST_GREEN, linewidth=3, solid_capstyle="round")
    ax.scatter([needle_angle], [0], color=theme.FOREST_GREEN, s=60, zorder=5)

    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, 1.3)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    fig.tight_layout()
    # Figure-coordinate (not polar-axes) text placement: polar axes only
    # accept non-negative radii, so the percentage label — which sits
    # below the arc, outside the plotted radius range — is added directly
    # to the figure instead of the polar axes.
    fig.text(
        0.5, 0.08, f"{risk_score * 100:.1f}%",
        ha="center", va="center", fontsize=22, fontweight="bold", color=theme.FOREST_GREEN,
    )
    return fig


def render_prediction_badge(result: AnalysisResult) -> None:
    if result.prediction_label == "Abnormality Detected":
        color, formal_label = theme.RISK_HIGH, "ABNORMALITY DETECTED"
    else:
        color, formal_label = theme.RISK_LOW, "NO ACUTE FINDINGS"

    st.markdown(
        f"""
        <div style="
            background: {color}1F;
            border: 1px solid {color}66;
            border-radius: 0.5rem;
            padding: 0.85rem 1.2rem;
            text-align: center;
            margin-bottom: 0.8rem;
        ">
            <span style="font-size: 1.3rem; font-weight: 700; letter-spacing: 0.02em; color: {color};">
                {formal_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_condition_risk_badge(label: str, value: Optional[float]) -> None:
    """Formal triage badge (LOW / MODERATE / HIGH) plus an approximate
    95% confidence interval for one condition's tear-risk probability —
    mirrors `qknee.ui.dashboard.render_risk_gauge`'s LOW/MODERATE/HIGH
    tiering (thresholds 0.33/0.66) so the two apps agree. Renders "N/A"
    when `value` is `None` (the condition's head isn't available for this
    backend/case)."""
    if value is None:
        st.metric(label=label, value="N/A")
        st.markdown('<span class="qknee-badge qknee-badge-neutral">UNAVAILABLE</span>', unsafe_allow_html=True)
        st.progress(0.0)
        return

    _, tier = theme.risk_tier(value)
    st.metric(label=label, value=f"{value * 100:.1f}%", delta=tier)
    st.markdown(theme.risk_badge_html(label, value), unsafe_allow_html=True)
    st.markdown(f'<div class="qknee-ci">{theme.format_confidence_interval(value)}</div>', unsafe_allow_html=True)
    st.progress(min(max(value, 0.0), 1.0))


def render_quantum_attribution_panel(pauli_z_expectations: Optional[np.ndarray]) -> Optional[plt.Figure]:
    """Quantum State Attribution Metrics panel: a formal bar chart of the
    4-qubit circuit's raw per-qubit Pauli-Z expectation values, each in
    `[-1.0, 1.0]` — the quantum circuit's own measurement output, mapping
    Hilbert-space rotations directly to feature impact, shown before the
    classical readout layer collapses it into one risk probability.

    Returns `None` (renders nothing) if `pauli_z_expectations` is `None`
    (the loaded VQC doesn't expose one — see `get_pauli_z_expectations`).
    """
    if pauli_z_expectations is None:
        return None

    n_qubits = len(pauli_z_expectations)
    fig, ax = plt.subplots(figsize=(4, 2.6))
    fig.patch.set_facecolor(theme.CARD_SURFACE)
    ax.set_facecolor(theme.CARD_SURFACE)

    colors = [theme.FOREST_GREEN if value >= 0 else theme.RISK_HIGH for value in pauli_z_expectations]
    qubit_labels = [f"Q{i}" for i in range(n_qubits)]
    ax.bar(qubit_labels, pauli_z_expectations, color=colors, edgecolor="none")
    ax.axhline(0.0, color=theme.TEXT_MUTED, linewidth=0.8)
    ax.grid(axis="y", color="#CBD5E1", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Pauli-Z Expectation ⟨Z⟩", color=theme.STERILE_WHITE)
    ax.set_title("Hilbert-Space Rotation Attribution", color=theme.FOREST_GREEN, fontsize=10)
    ax.tick_params(colors=theme.TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_color(theme.BORDER_GREY)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def render_header() -> None:
    st.set_page_config(
        page_title="Q-Knee Diagnostic Workstation",
        page_icon=theme.CLINICAL_GLYPH,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject_clinical_theme()
    theme.render_institutional_masthead(active_module="Diagnostic Workstation")
    theme.render_disclosure_banner()
    st.markdown(
        """
        <div class="qknee-eyebrow">PACS-Style Diagnostic Workstation</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Quantum-assisted ACL / medial meniscus / MCL tear-risk triage — quantitative lesion "
        "localization and attribution breakdown on a single-scan review console."
    )


def render_study_info_panel() -> None:
    """Left-panel structured patient/study metadata box (Patient ID,
    Age/Sex, Magnet Field, TE/TR) — de-identified placeholder values, since
    this workstation ingests anonymized research volumes with no real
    DICOM patient header attached."""
    st.sidebar.markdown("### Study Information")
    st.sidebar.markdown(
        f"""
        <div class="qknee-card" style="padding: 0.75rem 0.9rem;">
            <div class="qknee-mono" style="font-size: 0.78rem; line-height: 1.9; color: {theme.MONO_TEXT};">
                Patient ID: ANON-77201<br>
                Age/Sex: 34M<br>
                Magnet Field: 3.0T<br>
                TE/TR: 30/2800ms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[Optional[np.ndarray], str, float, float, str, float]:
    render_study_info_panel()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Radiological Ingestion Pipeline")
    uploaded_files = st.sidebar.file_uploader(
        "DICOM series (.dcm, select/drag multiple files), single DICOM, PNG, JPG, or NumPy volume (.npy)",
        type=["dcm", "dicom", "png", "jpg", "jpeg", "npy"],
        accept_multiple_files=True,
        help="Drag and drop one or more files — a multi-file .dcm selection is stacked into "
             "one 3D series, sorted by InstanceNumber/SliceLocation.",
    )

    runner = load_backend()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quantum Kernel Status")
    if runner is not None:
        st.sidebar.markdown(
            '<span class="qknee-badge qknee-badge-low">KERNEL ONLINE</span> '
            "Live PipelineRunner loaded (ResNet18 → PCA → 4-qubit VQC)",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<span class="qknee-badge qknee-badge-moderate">SIMULATION MODE</span> '
            "No trained backend found — deterministic simulation active",
            unsafe_allow_html=True,
        )
        error = st.session_state.get("_backend_error")
        if error:
            st.sidebar.caption(f"Reason: {error}")

    default_colormap, default_alpha = theme.DEFAULT_COLORMAP_NAME, 0.45

    if not uploaded_files:
        st.sidebar.info("Upload a study to enable plane/slice/contrast controls.")
        return None, PLANES[0], 0.5, 1.0, default_colormap, default_alpha

    volume = load_scan(uploaded_files)
    if volume is None:
        return None, PLANES[0], 0.5, 1.0, default_colormap, default_alpha

    display_name = (
        f"{len(uploaded_files)}-file DICOM series" if len(uploaded_files) > 1 else uploaded_files[0].name
    )
    st.sidebar.success(f"Loaded '{display_name}' — shape {volume.shape}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Viewport Controls")

    plane = st.sidebar.radio(
        "Primary Plane (quantum analysis target)", PLANES, horizontal=True,
        help="Which plane's slice is fed into the ResNet18 → PCA → VQC pipeline and Grad-CAM. "
             "All three planes are still shown, synchronized, below.",
    )

    # Shared slice-scrubbing slider: a single fractional position in [0, 1]
    # mapped independently onto each plane's own slice-count range, so ONE
    # slider keeps Sagittal/Coronal/Axial synchronized even though each has
    # a different depth along its own axis (see `_plane_slice_index` in
    # `main()`).
    slice_fraction = st.sidebar.slider(
        "Slice Depth Position (synced across all 3 planes)", 0.0, 1.0, 0.5, step=0.01,
        help="Scrubs the Axial, Coronal, and Sagittal viewports together — each plane maps this "
             "shared fractional position onto its own slice-depth range.",
    )

    contrast = st.sidebar.slider("Window Contrast", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Explainability Overlay Controls")

    colormap_names = list(theme.CLINICAL_COLORMAPS.keys())
    colormap_name = st.sidebar.selectbox(
        "Attribution Color Scale", colormap_names, index=colormap_names.index(default_colormap),
        help="Clinically standard color scale used to render the Grad-CAM attribution overlay.",
    )
    # Blends the raw Grad-CAM heatmap onto the MRI slice live via
    # `qknee.xai.gradcam.overlay_heatmap`'s `cv2.addWeighted(color_heatmap,
    # alpha, slice_bgr, 1 - alpha, 0)` call — a resize + colormap + weighted
    # sum, effectively free, so this slider re-blends in real time on every
    # drag with no re-inference. Discrete 5%-step percentage control (0%
    # to 100%), converted to the [0, 1] fraction `overlay_heatmap` expects.
    opacity_pct = st.sidebar.slider(
        "Attribution Overlay Opacity", min_value=0, max_value=100, value=int(default_alpha * 100), step=5,
        format="%d%%",
        help="Blend weight of the attribution overlay over the MRI slice (0% = slice only, "
             "100% = overlay only), applied live with no re-inference.",
    )
    alpha = opacity_pct / 100.0

    return volume, plane, slice_fraction, contrast, colormap_name, alpha


def _plane_slice_index(selector, plane: str, slice_fraction: float) -> Tuple[int, int]:
    """Maps the shared `slice_fraction` (`[0, 1]`) onto `plane`'s own
    slice-count range, so one fractional slider position keeps Axial /
    Coronal / Sagittal synchronized even though each has a different depth
    along its own axis. Returns `(index, max_index)`."""
    max_index = selector.num_slices(plane.lower()) - 1
    return round(slice_fraction * max_index), max_index


def render_synchronized_tri_plane_view(volume: np.ndarray, contrast: float, slice_fraction: float, primary_plane: str) -> None:
    """Synchronized Orthogonal Viewport: Sagittal, Coronal, and Axial
    rendered simultaneously with a PACS-style medical crosshair marking
    each plane's anatomical center, all driven by the one shared
    `slice_fraction` slider (see `_plane_slice_index`) — dragging that
    single sidebar control scrubs all three viewports together, instead
    of three independent per-plane sliders."""
    from qknee.data.ingestion import MultiPlaneViewSelector

    selector = MultiPlaneViewSelector(volume)
    st.markdown(f"#### {theme.icon('cube', size=17)} Synchronized Orthogonal Viewport", unsafe_allow_html=True)
    columns = st.columns(len(PLANES))
    for column, plane in zip(columns, PLANES):
        with column:
            index, max_index = _plane_slice_index(selector, plane, slice_fraction)
            is_primary = plane == primary_plane
            plane_slice = selector.get_slice(plane.lower(), index)
            plane_display = apply_contrast(normalize_uint8(plane_slice), contrast)
            plane_display = theme.draw_clinical_crosshair(plane_display)
            caption = theme.slice_depth_caption(plane, index, max_index, primary=is_primary)
            st.image(plane_display, channels="BGR", width="stretch", clamp=True, caption=caption)


def main() -> None:
    render_header()

    use_fast_path, fast_path_case = render_fast_path_sidebar()
    st.sidebar.markdown("---")
    if use_fast_path and fast_path_case is not None:
        render_fast_path_view(fast_path_case)
        render_deck_figures_expander()
        return

    volume, plane, slice_fraction, contrast, colormap_name, alpha = render_sidebar()

    if volume is None:
        st.info("Upload a `.png`, `.jpg`, `.npy`, or `.dcm` study (drag & drop supported) "
                 "from the sidebar to begin.")
        return

    from qknee.data.ingestion import MultiPlaneViewSelector

    colormap_value = theme.CLINICAL_COLORMAPS[colormap_name]

    selector = MultiPlaneViewSelector(volume)
    slice_index, max_index = _plane_slice_index(selector, plane, slice_fraction)
    raw_slice = selector.get_slice(plane.lower(), slice_index)
    display_slice = apply_contrast(normalize_uint8(raw_slice), contrast)

    result_key = "last_analysis_result"
    volumetric_key = "last_volumetric_result"

    st.markdown(
        f"#### {theme.icon('crosshair', size=17)} Quantitative Lesion Localization — Full-Volume Sweep",
        unsafe_allow_html=True,
    )
    vol_col1, vol_col2 = st.columns([1, 2])
    with vol_col1:
        run_volumetric_clicked = st.button(
            "Execute Full-Volume Attribution Sweep", width="stretch",
            help=f"Runs Grad-CAM independently on every slice of the {plane} plane "
                 f"(subsampled to at most {MAX_VOLUMETRIC_SLICES} slices for a deep volume) "
                 "and ranks them by salience — enables instant heatmap scrubbing below.",
        )
        if run_volumetric_clicked:
            with st.spinner(f"Running Grad-CAM across the {plane} volume..."):
                st.session_state[volumetric_key] = run_volumetric_gradcam(volume, plane)

    volumetric_result: Optional[VolumetricAnalysisResult] = st.session_state.get(volumetric_key)
    volumetric_active = volumetric_result is not None and volumetric_result.plane == plane

    with vol_col2:
        if volumetric_result is None:
            st.caption("Not yet run for this volume — click the button to analyze every slice.")
        elif not volumetric_active:
            st.caption(
                f"Cached result is for the **{volumetric_result.plane}** plane; "
                f"re-run for **{plane}** to enable synced scrubbing/top-3 highlighting here."
            )
        else:
            top3 = ", ".join(f"#{i + 1}" for i in volumetric_result.top_k_indices)
            st.caption(
                f"Analyzed {len(volumetric_result.analyzed_indices)} slice(s) of the {plane} plane "
                f"({volumetric_result.backend} backend). Top-3 high-salience slices: {top3 or '—'}."
            )
            if slice_index in volumetric_result.top_k_indices:
                rank = volumetric_result.top_k_indices.index(slice_index) + 1
                st.success(f"Slice {slice_index + 1} is a top-{rank} high-salience slice.")

    render_synchronized_tri_plane_view(volume, contrast, slice_fraction, primary_plane=plane)
    if volumetric_active and volumetric_result.top_k_indices:
        jump_labels = [f"Slice {i + 1}" for i in volumetric_result.top_k_indices]
        st.caption(
            f"Primary plane ({plane}) slice {slice_index + 1}/{max_index + 1} — jump to a high-salience "
            "slice using the sidebar's Slice Position control: " + ", ".join(jump_labels)
        )
    st.markdown("---")

    st.markdown(f"#### {theme.icon('cube', size=17)} Radiological Viewports", unsafe_allow_html=True)
    viewport_a_col, viewport_b_col, risk_col = st.columns([1, 1, 1.1])

    with viewport_a_col:
        st.markdown("##### Viewport A — Raw Ingestion Slice")
        oriented_slice = theme.draw_orientation_markers(display_slice, plane)
        st.image(
            oriented_slice, channels="BGR", width="stretch",
            caption=theme.slice_depth_caption(plane, slice_index, max_index, primary=True),
        )

    result: Optional[AnalysisResult] = st.session_state.get(result_key)

    with viewport_b_col:
        st.markdown("##### Viewport B — Grad-CAM Saliency Overlay")

        # Real-time synchronized overlay: prefer the volumetric result's
        # precomputed per-slice heatmap for the currently-scrubbed slice
        # (zero-recompute — just a resize + colormap + blend) over the
        # single-slice `result`, so dragging the sidebar's Slice control
        # instantly updates the heatmap once a volumetric run exists.
        volumetric_overlay = (
            volumetric_result.overlay_for(slice_index, display_slice, alpha=alpha, colormap=colormap_value)
            if volumetric_active else None
        )

        if volumetric_overlay is not None:
            st.image(
                volumetric_overlay, channels="BGR", width="stretch",
                caption=f"Slice {slice_index + 1} — synced from full-volume sweep "
                        f"({colormap_name}, opacity={alpha * 100:.0f}%)",
            )
        elif result is not None and result.gradcam_heatmap is not None:
            from qknee.xai.gradcam import overlay_heatmap

            overlay = overlay_heatmap(result.gradcam_heatmap, display_slice, alpha=alpha, colormap=colormap_value)
            st.image(overlay, channels="BGR", width="stretch", caption="Regions driving the predicted risk score")
        elif result is not None and result.gradcam_overlay is not None:
            st.image(
                result.gradcam_overlay,
                channels="BGR",
                width="stretch",
                caption="Regions driving the predicted risk score",
            )
        else:
            st.info("Execute the single-slice or full-volume analysis to generate an attribution overlay.")
        st.caption("Overlay opacity is adjustable from the sidebar's Explainability Overlay Controls.")

    with risk_col:
        st.markdown("##### Quantitative Risk Assessment")
        run_clicked = st.button("Execute Diagnostic Inference", type="primary", width="stretch")

        if run_clicked:
            with st.spinner("Executing radiological ingestion pipeline and quantum kernel..."):
                st.session_state[result_key] = run_analysis(raw_slice)

        result = st.session_state.get(result_key)

        if result is None:
            st.info("Select **Execute Diagnostic Inference** to generate a prediction for this slice.")
        else:
            render_prediction_badge(result)
            render_condition_risk_badge("ACL Tear Probability", result.acl_risk)
            render_condition_risk_badge("Meniscal Tear Probability", result.meniscus_risk)
            render_condition_risk_badge("MCL Sprain Probability", result.mcl_risk)
            st.metric("Quantum Circuit Latency", format_latency_ms(result.quantum_latency_ms))
            st.metric("Total Pipeline Latency", format_latency_ms(result.total_latency_ms))
            st.caption(f"Backend: **{result.backend}**")

    st.markdown("---")
    st.markdown(f"#### {theme.icon('circuit', size=17)} Quantum State Attribution Metrics", unsafe_allow_html=True)
    result = st.session_state.get(result_key)

    chart_col, action_col = st.columns([1.3, 1])
    with chart_col:
        st.markdown("##### 4-Qubit Pauli-Z Expectation Vector")
        pauli_z = result.pauli_z_expectations if result is not None else None
        fig = render_quantum_attribution_panel(pauli_z)
        if fig is not None:
            st.pyplot(fig, width="stretch")
            st.caption("Per-qubit Pauli-Z expectation ⟨Z⟩ ∈ [-1.0, 1.0], mapping Hilbert-space rotations "
                       "directly to feature impact, read before the classical readout layer.")
        else:
            st.info("Execute **Diagnostic Inference** to display this slice's per-qubit ⟨Z⟩ expectation values.")

    with action_col:
        st.markdown("##### Action Bar")
        if result is not None:
            render_report_download(display_slice, result)
            st.button("Sign & Lock Study", width="stretch",
                      help="Locks this study's diagnostic session under the reviewing radiologist's "
                           "attestation. Confirmatory over-read is required before clinical release.")
        else:
            st.info("Execute **Diagnostic Inference** to enable report export and study sign-off.")

    render_deck_figures_expander()

    st.markdown("---")
    st.caption(
        "Radiological Ingestion Pipeline: upload → contrast-adjusted preview → ResNet18 (512-D) → "
        "PCA → [0, 2π] angle scaling → 4-qubit PennyLane VQC → risk score. "
        "Grad-CAM is backpropagated from the risk score itself (not embedding energy), "
        "so the overlay explains that prediction. " + theme.NOT_A_DEVICE_FOOTNOTE
    )


if __name__ == "__main__":
    main()
