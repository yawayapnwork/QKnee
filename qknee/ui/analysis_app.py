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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

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

# The two reference planes always shown for anatomical cross-checking
# alongside whichever plane is primary (task spec names these two
# specifically, regardless of which plane the main slice explorer is on).
CROSS_REFERENCE_PLANES = ["Sagittal", "Coronal"]

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
    risk_score: float
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

    Returns `None` (rather than guessing) if `vqc_model` doesn't expose a
    `.quantum_layer` attribute — e.g. a custom ansatz that doesn't follow
    `VQCClassifier`'s structure.
    """
    quantum_layer = getattr(vqc_model, "quantum_layer", None)
    if quantum_layer is None:
        return None

    import torch

    with torch.no_grad():
        angles_tensor = torch.from_numpy(np.asarray(quantum_angles)).float()
        expvals = quantum_layer(angles_tensor)
    return expvals.detach().cpu().numpy().reshape(-1)


# --------------------------------------------------------------------------- #
# Backend loading + inference (mock-fallback pattern shared with app.py)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_backend():
    """Loads a `PipelineRunner` (DataIngestion -> ResNet18 -> PCA -> VQC ->
    GradCAM) if a fitted PCA artifact is available; returns None otherwise
    so the UI can fall back to a deterministic mock."""
    if not _config.paths.pca_artifact.exists():
        return None

    try:
        from qknee.models.pipeline import PipelineRunner

        return PipelineRunner(config=_config)
    except Exception as exc:  # noqa: BLE001
        st.session_state["_backend_error"] = str(exc)
        return None


@st.cache_resource(show_spinner=False)
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
    """Renders the 'Judge Mode' sidebar toggle that bypasses upload/live/mock
    inference entirely and replays one of `precomputed_cache.json`'s
    pre-scored cases straight from process memory. Returns
    `(use_fast_path, selected_case)`."""
    st.sidebar.markdown("### 🏎️ Judge Mode")

    cache = load_precomputed_cache()
    cases: List[dict] = (cache or {}).get("cases", [])

    use_fast_path = st.sidebar.toggle(
        "Enable NISQ Simulation Fast-Path (0-latency Cache)",
        value=False,
        disabled=not cases,
        help=f"Instantly replays one of {len(cases)} precomputed case(s) from "
             f"`{PRECOMPUTED_CACHE_PATH}` — zero model load, zero QNode "
             "execution — for judge-facing demos where live inference "
             "latency risks a UI timeout.",
    )

    if not cases:
        st.sidebar.caption(
            f"No precomputed cache found at `{PRECOMPUTED_CACHE_PATH}`. Run "
            "`python scripts/generate_demo_cache.py` to build one."
        )
        return False, None
    if not use_fast_path:
        st.sidebar.caption(f"{len(cases)} fast-path case(s) available.")
        return False, None

    case_labels = [f"{case['case_id']} ({case.get('plane', '?')})" for case in cases]
    selected_label = st.sidebar.selectbox("Fast-Path Case", case_labels)
    selected_case = cases[case_labels.index(selected_label)]
    st.sidebar.success(f"⚡ Serving '{selected_case['case_id']}' from cache — 0 ms inference.")
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
    )
    return overlay, result


def render_fast_path_view(case: dict) -> None:
    """Renders a lightweight, fully self-contained view of one Judge
    Fast-Path case — bypasses upload/plane/slice controls entirely, since
    the cached case carries its own precomputed image/scores."""
    st.info(
        f"⚡ Judge Fast-Path active — serving precomputed case **{case['case_id']}** "
        "from cache (0 ms inference)."
    )
    overlay, result = build_fast_path_result(case)

    image_col, gauge_col, attrib_col = st.columns([1, 1, 1])

    with image_col:
        st.markdown(f"#### Precomputed Grad-CAM — {case.get('plane', '?').title()} plane")
        if overlay is not None:
            snippet = case.get("clinical_text_snippet")
            caption = snippet.splitlines()[0] if snippet else None
            st.image(overlay, channels="BGR", use_container_width=True, caption=caption)
        else:
            st.warning("No cached heatmap available for this case.")

    with gauge_col:
        render_prediction_badge(result)
        st.pyplot(render_risk_gauge(result.risk_score), use_container_width=True)
        st.metric("Quantum Circuit Latency (cached)", f"{result.quantum_latency_ms:.1f} ms")
        st.caption(f"Backend: **{result.backend}**")

    with attrib_col:
        fig = render_quantum_attribution_panel(result.pauli_z_expectations)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.caption("Per-qubit Pauli-Z expectation ⟨Z⟩, precomputed offline for this case.")


def render_deck_figures_expander() -> None:
    """Surfaces the offline-generated deck figures (`scripts/generate_deck_assets.py`)
    — e.g. the circuit diagram — inline for reference, when available."""
    figure_path = DECK_FIGURES_DIR / "circuit_diagram.png"
    if not figure_path.exists():
        return
    with st.expander("🧬 Quantum Circuit Diagram (reference)"):
        st.image(str(figure_path), use_container_width=True)


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

    t0 = time.perf_counter()
    batch = runner.ingest(slice_2d)
    features = runner.extract_resnet_features(batch)
    quantum_angles = runner.reduce_to_quantum_angles(features)
    t1 = time.perf_counter()
    risk_score = runner.classify(quantum_angles)
    t2 = time.perf_counter()

    gradcam_overlay: Optional[np.ndarray] = None
    gradcam_heatmap: Optional[np.ndarray] = None
    try:
        gradcam_heatmap = runner.explain(batch[:, 0])
        gradcam_overlay = overlay_heatmap(gradcam_heatmap, slice_2d)
    except Exception as exc:  # noqa: BLE001 - a failed heatmap shouldn't hide the risk score
        logger.warning("Grad-CAM generation failed; showing risk score without an overlay: %s", exc)

    quantum_latency_ms = (t2 - t1) * 1000
    total_latency_ms = (t2 - t0) * 1000
    label = "Abnormality Detected" if risk_score >= RISK_THRESHOLD else "Normal"
    pauli_z_expectations = get_pauli_z_expectations(runner.vqc, quantum_angles)

    return AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=quantum_latency_ms,
        total_latency_ms=total_latency_ms,
        backend="live",
        gradcam_overlay=gradcam_overlay,
        gradcam_heatmap=gradcam_heatmap,
        pauli_z_expectations=pauli_z_expectations,
    )


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

    return VolumetricAnalysisResult(
        plane=plane,
        heatmaps={s.slice_index: s.heatmap for s in result.saliencies},
        magnitudes={s.slice_index: s.magnitude for s in result.saliencies},
        top_k_indices=result.top_k_indices,
        analyzed_indices=indices,
        backend="live",
    )


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

def load_scan(uploaded_files: List) -> Optional[np.ndarray]:
    """Loads one or more drag-and-dropped uploads into a `(D, H, W)`
    volume, via `qknee.data.ingestion.DataIngestion.load_volume_array` —
    the same DICOM-series/.npy/single-DICOM loading path
    `qknee.ui.dashboard` uses, so a multi-file `.dcm` series (one file per
    slice) stacks into a real tri-planar volume here too, not just a flat
    single image.

    Args:
        uploaded_files: One or more Streamlit `UploadedFile`s (`.png`,
            `.jpg`, `.jpeg`, `.npy`, or `.dcm`/`.dicom` — multiple `.dcm`
            files are treated as one series, sorted and stacked by
            `InstanceNumber`/`SliceLocation`).

    Returns `None` (and shows a Streamlit error) if the upload can't be parsed.
    """
    from qknee.data.ingestion import DataIngestion, IngestionError

    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    if not files:
        return None

    source = files if len(files) > 1 else files[0]
    display_name = f"{len(files)}-file DICOM series" if len(files) > 1 else files[0].name

    try:
        array = DataIngestion().load_volume_array(source)
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
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    zones = [(0.0, 0.33, "#2ECC71"), (0.33, 0.66, "#F5A623"), (0.66, 1.0, "#E74C3C")]
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
    ax.plot([needle_angle, needle_angle], [0, 1.15], color="white", linewidth=3, solid_capstyle="round")
    ax.scatter([needle_angle], [0], color="white", s=60, zorder=5)

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
        ha="center", va="center", fontsize=22, fontweight="bold", color="white",
    )
    return fig


def render_prediction_badge(result: AnalysisResult) -> None:
    if result.prediction_label == "Abnormality Detected":
        color, icon = "#E74C3C", "⚠️"
    else:
        color, icon = "#2ECC71", "✅"

    st.markdown(
        f"""
        <div style="
            background: {color}22;
            border: 1px solid {color};
            border-radius: 0.6rem;
            padding: 0.9rem 1.2rem;
            text-align: center;
            margin-bottom: 0.8rem;
        ">
            <span style="font-size: 1.6rem; font-weight: 700; color: {color};">
                {icon} {result.prediction_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quantum_attribution_panel(pauli_z_expectations: Optional[np.ndarray]) -> Optional[plt.Figure]:
    """Quantum State Attribution panel: a bar chart of the 4-qubit
    circuit's raw per-qubit Pauli-Z expectation values, each in
    `[-1.0, 1.0]` — the quantum circuit's own measurement output, shown
    before the classical readout layer collapses it into one risk
    probability, so a viewer can see how each qubit individually
    contributed to the decision rather than only the final score.

    Returns `None` (renders nothing) if `pauli_z_expectations` is `None`
    (the loaded VQC doesn't expose one — see `get_pauli_z_expectations`).
    """
    if pauli_z_expectations is None:
        return None

    n_qubits = len(pauli_z_expectations)
    fig, ax = plt.subplots(figsize=(4, 2.6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    colors = ["#2ECC71" if value >= 0 else "#E74C3C" for value in pauli_z_expectations]
    qubit_labels = [f"q{i}" for i in range(n_qubits)]
    ax.bar(qubit_labels, pauli_z_expectations, color=colors, edgecolor="none")
    ax.axhline(0.0, color="#8B949E", linewidth=0.8)

    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("⟨Z⟩", color="#E6EDF3")
    ax.tick_params(colors="#E6EDF3")
    for spine in ax.spines.values():
        spine.set_color("#8B949E")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def render_header() -> None:
    st.set_page_config(
        page_title="Q-Knee Analysis",
        page_icon="🦵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background-color: #0E1117; }
        .qknee-title { font-size: 2rem; font-weight: 800; margin-bottom: 0; }
        .qknee-subtitle { color: #8B949E; font-size: 0.85rem; margin-top: -0.3rem; }
        </style>
        <div class="qknee-title">🦵 Q-Knee Analysis</div>
        <div class="qknee-subtitle">
            Quantum-assisted ACL/meniscal tear screening — research prototype, not for clinical use.
        </div>
        <br/>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[Optional[np.ndarray], str, int, float, str, float]:
    st.sidebar.markdown("### 📤 Upload Scan")
    uploaded_files = st.sidebar.file_uploader(
        "DICOM series (.dcm, select/drag multiple files), single DICOM, PNG, JPG, or NumPy volume (.npy)",
        type=["dcm", "dicom", "png", "jpg", "jpeg", "npy"],
        accept_multiple_files=True,
        help="Drag and drop one or more files — a multi-file .dcm selection is stacked into "
             "one 3D series, sorted by InstanceNumber/SliceLocation.",
    )

    runner = load_backend()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚛️ Backend Status")
    if runner is not None:
        st.sidebar.markdown("🟢 **Live PipelineRunner loaded** (ResNet18 → PCA → 4-qubit VQC)")
    else:
        st.sidebar.markdown("🟡 **Mock mode** — no trained backend found")
        error = st.session_state.get("_backend_error")
        if error:
            st.sidebar.caption(f"Reason: {error}")

    default_colormap, default_alpha = "jet", 0.45

    if not uploaded_files:
        st.sidebar.info("Upload a scan to enable plane/slice/contrast controls.")
        return None, PLANES[0], 0, 1.0, default_colormap, default_alpha

    volume = load_scan(uploaded_files)
    if volume is None:
        return None, PLANES[0], 0, 1.0, default_colormap, default_alpha

    display_name = (
        f"{len(uploaded_files)}-file DICOM series" if len(uploaded_files) > 1 else uploaded_files[0].name
    )
    st.sidebar.success(f"Loaded '{display_name}' — shape {volume.shape}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎚️ View Controls")

    plane = st.sidebar.radio(
        "Anatomical Plane", PLANES, horizontal=True,
        help="Scroll through the loaded volume in any of the three anatomical planes.",
    )

    from qknee.data.ingestion import MultiPlaneViewSelector

    max_index = MultiPlaneViewSelector(volume).num_slices(plane.lower()) - 1
    if max_index > 0:
        slice_index = st.sidebar.slider("Slice", 0, max_index, max_index // 2)
    else:
        slice_index = 0
        st.sidebar.caption("Single-slice image — slider disabled.")

    contrast = st.sidebar.slider("Contrast", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Grad-CAM Display")
    from qknee.xai.gradcam import COLORMAP_OPTIONS

    colormap_name = st.sidebar.selectbox(
        "Colormap", list(COLORMAP_OPTIONS.keys()), index=list(COLORMAP_OPTIONS.keys()).index(default_colormap),
        help="Color scheme used to render the Grad-CAM heatmap.",
    )
    # Blends the raw Grad-CAM heatmap onto the MRI slice live via
    # `qknee.xai.gradcam.overlay_heatmap`'s `cv2.addWeighted(color_heatmap,
    # alpha, slice_bgr, 1 - alpha, 0)` call — a resize + colormap + weighted
    # sum, effectively free, so this slider re-blends in real time on every
    # drag with no re-inference.
    alpha = st.sidebar.slider(
        "Heatmap Overlay Opacity", 0.0, 1.0, default_alpha, step=0.05,
        help="Blend weight of the heatmap over the MRI slice, applied live via cv2.addWeighted "
             "(0 = only the slice, 1 = only the heatmap).",
    )

    return volume, plane, slice_index, contrast, colormap_name, alpha


def render_cross_reference_panel(volume: np.ndarray, contrast: float, primary_plane: str) -> None:
    """Dual-plane anatomical cross-referencing: Sagittal + Coronal
    thumbnails, each independently scrubbable (their own slider, keyed so
    Streamlit persists each plane's position across reruns), so a viewer
    can confirm the primary view's anatomical location on the two
    complementary planes. Always shows both, regardless of which plane is
    currently primary."""
    from qknee.data.ingestion import MultiPlaneViewSelector

    selector = MultiPlaneViewSelector(volume)
    st.markdown("#### 🧭 Dual-Plane Cross-Reference")
    columns = st.columns(len(CROSS_REFERENCE_PLANES))
    for column, ref_plane in zip(columns, CROSS_REFERENCE_PLANES):
        with column:
            ref_max_index = selector.num_slices(ref_plane.lower()) - 1
            is_primary = ref_plane == primary_plane
            label = f"{ref_plane} slice" + (" (primary view)" if is_primary else "")
            if ref_max_index > 0:
                ref_index = st.slider(
                    label, 0, ref_max_index, ref_max_index // 2, key=f"crossref_slice_{ref_plane}",
                )
            else:
                ref_index = 0
                st.caption(f"{label}: single slice.")
            ref_slice = selector.get_slice(ref_plane.lower(), ref_index)
            ref_display = apply_contrast(normalize_uint8(ref_slice), contrast)
            st.image(ref_display, use_container_width=True, clamp=True, caption=f"{ref_plane} — {ref_index + 1}/{ref_max_index + 1}")


def main() -> None:
    render_header()

    use_fast_path, fast_path_case = render_fast_path_sidebar()
    st.sidebar.markdown("---")
    if use_fast_path and fast_path_case is not None:
        render_fast_path_view(fast_path_case)
        render_deck_figures_expander()
        return

    volume, plane, slice_index, contrast, colormap_name, alpha = render_sidebar()

    if volume is None:
        st.info("👈 Upload a `.png`, `.jpg`, `.npy`, or `.dcm` scan (drag & drop supported) "
                 "from the sidebar to begin.")
        return

    from qknee.data.ingestion import MultiPlaneViewSelector
    from qknee.xai.gradcam import COLORMAP_OPTIONS

    colormap_value = COLORMAP_OPTIONS[colormap_name]

    selector = MultiPlaneViewSelector(volume)
    max_index = selector.num_slices(plane.lower()) - 1
    raw_slice = selector.get_slice(plane.lower(), slice_index)
    display_slice = apply_contrast(normalize_uint8(raw_slice), contrast)

    result_key = "last_analysis_result"
    volumetric_key = "last_volumetric_result"

    st.markdown("#### 🧠 Multi-Slice Grad-CAM (full volume)")
    vol_col1, vol_col2 = st.columns([1, 2])
    with vol_col1:
        run_volumetric_clicked = st.button(
            "Run Volumetric Grad-CAM Analysis", use_container_width=True,
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
                st.success(f"⭐ Slice {slice_index + 1} is a top-{rank} high-salience slice.")

    image_col, gradcam_col, action_col = st.columns([1, 1, 1.2])

    with image_col:
        st.markdown(f"#### {plane} View — Slice {slice_index + 1} / {max_index + 1}")
        st.image(display_slice, use_container_width=True, clamp=True)
        if volumetric_active and volumetric_result.top_k_indices:
            jump_labels = [f"Slice {i + 1}" for i in volumetric_result.top_k_indices]
            st.caption("Jump to a high-salience slice using the sidebar's Slice control: " + ", ".join(jump_labels))

    with action_col:
        st.markdown("#### Run Analysis")
        st.caption("Runs ResNet18 feature extraction + the 4-qubit quantum classifier on the slice above.")
        run_clicked = st.button("🔬 Run Q-Knee Analysis", type="primary", use_container_width=True)

        if run_clicked:
            with st.spinner("Extracting features and running the quantum circuit..."):
                st.session_state[result_key] = run_analysis(raw_slice)

        result: Optional[AnalysisResult] = st.session_state.get(result_key)

        if result is None:
            st.info("Click **Run Q-Knee Analysis** to generate a prediction for this slice.")
        else:
            render_prediction_badge(result)
            st.pyplot(render_risk_gauge(result.risk_score), use_container_width=True)
            st.metric("Quantum Circuit Latency", f"{result.quantum_latency_ms:.1f} ms")
            st.metric("Total Pipeline Latency", f"{result.total_latency_ms:.1f} ms")
            st.caption(f"Backend: **{result.backend}**")

    with gradcam_col:
        st.markdown("#### Grad-CAM")
        result = st.session_state.get(result_key)

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
                volumetric_overlay, channels="BGR", use_container_width=True,
                caption=f"Slice {slice_index + 1} — synced from volumetric analysis ({colormap_name}, α={alpha:.2f})",
            )
        elif result is not None and result.gradcam_heatmap is not None:
            from qknee.xai.gradcam import overlay_heatmap

            overlay = overlay_heatmap(result.gradcam_heatmap, display_slice, alpha=alpha, colormap=colormap_value)
            st.image(overlay, channels="BGR", use_container_width=True, caption="Regions driving the risk prediction")
        elif result is not None and result.gradcam_overlay is not None:
            st.image(
                result.gradcam_overlay,
                channels="BGR",
                use_container_width=True,
                caption="Regions driving the risk prediction",
            )
        else:
            st.info("Run the single-slice or volumetric analysis to generate a Grad-CAM overlay.")

    st.markdown("---")
    attribution_col, crossref_col = st.columns([1, 2])
    with attribution_col:
        st.markdown("#### ⚛️ Quantum State Attribution")
        result = st.session_state.get(result_key)
        pauli_z = result.pauli_z_expectations if result is not None else None
        fig = render_quantum_attribution_panel(pauli_z)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.caption("Per-qubit Pauli-Z expectation ⟨Z⟩, read directly from the quantum circuit "
                       "before the classical readout layer.")
        else:
            st.info("Run **Q-Knee Analysis** to display this slice's per-qubit ⟨Z⟩ expectation values.")

    with crossref_col:
        render_cross_reference_panel(volume, contrast, primary_plane=plane)

    render_deck_figures_expander()

    st.markdown("---")
    st.caption(
        "Pipeline: upload → contrast-adjusted preview → ResNet18 (512-D) → "
        "PCA → [0, 2π] angle scaling → 4-qubit PennyLane VQC → risk score. "
        "Grad-CAM is backpropagated from the risk score itself (not embedding energy), "
        "so the heatmap explains that prediction."
    )


if __name__ == "__main__":
    main()
