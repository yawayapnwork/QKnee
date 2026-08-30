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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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

PLANES = ["Axial", "Coronal", "Sagittal"]


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

    return AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=quantum_latency_ms,
        total_latency_ms=total_latency_ms,
        backend="live",
        gradcam_overlay=gradcam_overlay,
        gradcam_heatmap=gradcam_heatmap,
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

    return AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=quantum_latency_ms,
        total_latency_ms=total_latency_ms,
        backend="mock",
        gradcam_overlay=overlay_heatmap(heatmap, slice_2d),
        gradcam_heatmap=heatmap,
    )


def run_analysis(slice_2d: np.ndarray) -> AnalysisResult:
    runner = load_backend()
    if runner is not None:
        return run_live_analysis(runner, slice_2d)
    return run_mock_analysis(slice_2d)


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


def render_sidebar() -> Tuple[Optional[np.ndarray], str, int, float]:
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

    if not uploaded_files:
        st.sidebar.info("Upload a scan to enable plane/slice/contrast controls.")
        return None, PLANES[0], 0, 1.0

    volume = load_scan(uploaded_files)
    if volume is None:
        return None, PLANES[0], 0, 1.0

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

    return volume, plane, slice_index, contrast


def main() -> None:
    render_header()
    volume, plane, slice_index, contrast = render_sidebar()

    if volume is None:
        st.info("👈 Upload a `.png`, `.jpg`, `.npy`, or `.dcm` scan (drag & drop supported) "
                 "from the sidebar to begin.")
        return

    from qknee.data.ingestion import MultiPlaneViewSelector

    selector = MultiPlaneViewSelector(volume)
    max_index = selector.num_slices(plane.lower()) - 1
    raw_slice = selector.get_slice(plane.lower(), slice_index)
    display_slice = apply_contrast(normalize_uint8(raw_slice), contrast)

    image_col, gradcam_col, action_col = st.columns([1, 1, 1.2])

    with image_col:
        st.markdown(f"#### {plane} View — Slice {slice_index + 1} / {max_index + 1}")
        st.image(display_slice, use_container_width=True, clamp=True)

    result_key = "last_analysis_result"

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
        if result is None:
            st.info("Run the analysis to generate a Grad-CAM overlay.")
        elif result.gradcam_heatmap is not None:
            # Live opacity re-blend: overlay_heatmap is just a resize +
            # colormap + cv2.addWeighted, so re-blending on every slider
            # drag is effectively free — no re-inference required.
            from qknee.xai.gradcam import overlay_heatmap

            opacity = st.slider(
                "Heatmap Opacity", min_value=0.0, max_value=1.0,
                value=float(_config.gradcam.alpha), step=0.05,
                help="Blends the raw Grad-CAM heatmap onto the slice live.",
            )
            overlay = overlay_heatmap(result.gradcam_heatmap, display_slice, alpha=opacity)
            st.image(overlay, channels="BGR", use_container_width=True, caption="Regions driving the risk prediction")
        elif result.gradcam_overlay is not None:
            st.image(
                result.gradcam_overlay,
                channels="BGR",
                use_container_width=True,
                caption="Regions driving the risk prediction",
            )
        else:
            st.info("Grad-CAM overlay unavailable for this slice.")

    st.markdown("---")
    st.caption(
        "Pipeline: upload → contrast-adjusted preview → ResNet18 (512-D) → "
        "PCA → [0, 2π] angle scaling → 4-qubit PennyLane VQC → risk score. "
        "Grad-CAM is backpropagated from the risk score itself (not embedding energy), "
        "so the heatmap explains that prediction."
    )


if __name__ == "__main__":
    main()
