"""
Q-Knee: interactive clinical diagnostic dashboard (Streamlit).

Lets a clinician/researcher upload a DICOM or .npy MRI volume, scroll through
slices in any of the three anatomical planes, and see real-time ACL/meniscus
tear-risk scores produced by the ResNet18 -> PCA -> 4-qubit VQC pipeline
built in `qknee/data`, `qknee/models`, and `qknee/xai`, orchestrated by
`qknee.models.pipeline.PipelineRunner`.

Backend priority: the HTTP API (`$QKNEE_API_URL`, when set and reachable) is
queried first, per the two-service docker-compose architecture; if that's
unavailable, an in-process `PipelineRunner` is used; if the PCA artifact
isn't fitted either, the dashboard falls back to a seeded mock inference
engine, so the UI remains fully demoable in every case.

Run with:
    streamlit run qknee/ui/dashboard.py

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

# --------------------------------------------------------------------------- #
# Backend wiring: HTTP API (preferred, when $QKNEE_API_URL is set and
# reachable) -> in-process PipelineRunner -> seeded mock, in that order.
# --------------------------------------------------------------------------- #

_config = load_config()
logger = get_logger(__name__)


@dataclass
class InferenceResult:
    acl_risk: float                    # [0, 1]
    meniscus_risk: Optional[float]     # [0, 1], or None when unavailable (e.g. "api" backend)
    resnet_latency_ms: float
    pca_latency_ms: float
    quantum_latency_ms: float
    total_latency_ms: float
    backend: str             # "live", "mock", or "api"
    gradcam_overlay: Optional[np.ndarray] = None  # (H, W, 3) BGR uint8, or None if generation failed


@st.cache_resource(show_spinner=False)
def load_backend() -> Tuple[Optional[object], Optional[object], Optional[object]]:
    """Attempts to load the real ResNet18 -> PCA -> VQC pipeline.

    Returns a tuple (runner, acl_model, meniscus_model) or (None, None, None)
    if any dependency (torch, pennylane, or the fitted PCA artifact) is
    unavailable — the caller falls back to mock inference in that case.
    """
    if not _config.paths.pca_artifact.exists():
        return None, None, None

    try:
        import torch

        from qknee.models.pipeline import PipelineRunner, PipelineValidationError, load_vqc_weights
        from qknee.models.vqc import VQCClassifier

        runner = PipelineRunner(config=_config)

        # Two independent quantum heads: one scored for ACL tear risk, one
        # for meniscus tear risk. Each loads its own trained checkpoint from
        # config.yaml's paths.acl_checkpoint / paths.meniscus_checkpoint when
        # available; a missing/invalid checkpoint falls back to randomly
        # initialized weights for that head only (not the whole backend).
        torch.manual_seed(42)
        acl_model = VQCClassifier()
        if _config.paths.acl_checkpoint.exists():
            try:
                load_vqc_weights(acl_model, _config.paths.acl_checkpoint)
                logger.info("Loaded trained ACL VQC weights from %s", _config.paths.acl_checkpoint)
            except PipelineValidationError as exc:
                logger.warning("Failed to load ACL checkpoint (%s); using random weights: %s",
                                _config.paths.acl_checkpoint, exc)
        else:
            logger.warning("No ACL checkpoint found at %s; using randomly initialized weights.",
                            _config.paths.acl_checkpoint)
        acl_model.eval()

        torch.manual_seed(7)
        meniscus_model = VQCClassifier()
        if _config.paths.meniscus_checkpoint.exists():
            try:
                load_vqc_weights(meniscus_model, _config.paths.meniscus_checkpoint)
                logger.info("Loaded trained meniscus VQC weights from %s", _config.paths.meniscus_checkpoint)
            except PipelineValidationError as exc:
                logger.warning("Failed to load meniscus checkpoint (%s); using random weights: %s",
                                _config.paths.meniscus_checkpoint, exc)
        else:
            logger.warning("No meniscus checkpoint found at %s; using randomly initialized weights.",
                            _config.paths.meniscus_checkpoint)
        meniscus_model.eval()

        return runner, acl_model, meniscus_model
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as "unavailable"
        st.session_state.setdefault("_backend_error", str(exc))
        return None, None, None


# --------------------------------------------------------------------------- #
# HTTP API client (preferred backend when $QKNEE_API_URL is reachable)
# --------------------------------------------------------------------------- #

def resolve_api_url() -> Optional[str]:
    """Reads the FastAPI backend's URL from `$QKNEE_API_URL` (set for the
    `ui` container in docker-compose.yml). Returns None if unset, in
    which case the dashboard falls back to in-process/mock inference."""
    return os.environ.get("QKNEE_API_URL") or None


def api_is_reachable(api_url: str, timeout: float = 1.5) -> bool:
    """Cheap reachability probe against the API's `/health` endpoint.

    Returns False (never raises) on any connection error, timeout, or
    non-2xx response, so a down/unreachable API always degrades to the
    in-process/mock path rather than hanging the UI.
    """
    try:
        import requests

        response = requests.get(f"{api_url}/health", timeout=timeout)
        return response.status_code == 200
    except Exception as exc:  # noqa: BLE001 - any failure just means "not reachable"
        logger.debug("API health check failed for %s: %s", api_url, exc)
        return False


def run_api_inference(slice_2d: np.ndarray, api_url: str) -> InferenceResult:
    """Delegates inference to the Q-Knee FastAPI backend over HTTP
    (`POST {api_url}/predict`) instead of running `PipelineRunner`
    in-process — the two-service (api + ui) docker-compose
    architecture's intended data path.

    The API's `QKneeModel` exposes one unified risk score (no separate
    ACL/meniscus heads), so `meniscus_risk` is left `None` here rather than
    fabricating a second score; the UI renders that gauge as "N/A" in API mode.
    Per-stage latency isn't reported by the API either, so only the measured
    HTTP round-trip is attributed, to `total_latency_ms`.

    Raises on any HTTP/connection failure — callers should catch and fall
    back to `run_live_inference`/`run_mock_inference`.
    """
    import base64
    import io

    import cv2
    import requests

    buffer = io.BytesIO()
    np.save(buffer, slice_2d)

    t0 = time.perf_counter()
    response = requests.post(
        f"{api_url}/predict",
        files={"file": ("slice.npy", buffer.getvalue(), "application/octet-stream")},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    total_latency_ms = (time.perf_counter() - t0) * 1000

    gradcam_overlay: Optional[np.ndarray] = None
    heatmap_b64 = payload.get("gradcam_heatmap")
    if heatmap_b64:
        try:
            png_bytes = base64.b64decode(heatmap_b64)
            gradcam_overlay = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001 - a bad heatmap payload shouldn't hide the risk score
            logger.warning("Failed to decode Grad-CAM heatmap from API response: %s", exc)

    return InferenceResult(
        acl_risk=float(payload["risk_score"]),
        meniscus_risk=None,
        resnet_latency_ms=0.0,
        pca_latency_ms=0.0,
        quantum_latency_ms=0.0,
        total_latency_ms=total_latency_ms,
        backend=f"api/{payload.get('backend', 'unknown')}",
        gradcam_overlay=gradcam_overlay,
    )


def _seed_from_slice(slice_2d: np.ndarray) -> int:
    """Deterministic seed derived from slice content, so mock scores stay
    stable for the same slice/view rather than flickering on every rerun."""
    digest = hashlib.sha256(slice_2d.tobytes()).digest()
    return int.from_bytes(digest[:4], "big")


def _mock_gradcam_overlay(slice_2d: np.ndarray) -> np.ndarray:
    """Cheap, torch-free stand-in Grad-CAM overlay for mock mode (a soft
    radial gradient blended onto the slice via OpenCV only), so the heatmap
    panel is exercised in the UI even without a live backend — mirrors
    `qknee.api.server.QKneeBackend._predict_mock`'s fallback heatmap."""
    import cv2

    display = normalize_for_display(slice_2d)
    height, width = display.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2, width / 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    fake_heatmap = np.clip(1 - radius / radius.max(), 0, 1)

    gray_bgr = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    heatmap_uint8 = (fake_heatmap * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(color_heatmap, 0.45, gray_bgr, 0.55, 0)


def run_mock_inference(slice_2d: np.ndarray) -> InferenceResult:
    """Seeded, deterministic mock scores + plausible latency numbers, used
    whenever the real pipeline/model backend isn't available."""
    rng = np.random.default_rng(_seed_from_slice(slice_2d))

    resnet_ms = float(rng.uniform(18, 35))
    pca_ms = float(rng.uniform(0.5, 2.0))
    quantum_ms = float(rng.uniform(4, 12))

    return InferenceResult(
        acl_risk=float(rng.uniform(0.05, 0.95)),
        meniscus_risk=float(rng.uniform(0.05, 0.95)),
        resnet_latency_ms=resnet_ms,
        pca_latency_ms=pca_ms,
        quantum_latency_ms=quantum_ms,
        total_latency_ms=resnet_ms + pca_ms + quantum_ms,
        backend="mock",
        gradcam_overlay=_mock_gradcam_overlay(slice_2d),
    )


def run_live_inference(slice_2d: np.ndarray, runner, acl_model, meniscus_model) -> InferenceResult:
    """Runs the real DataIngestion -> ResNet18 -> PCA -> VQC pipeline (via
    `PipelineRunner`'s stage methods) on one 2D slice, timing each stage,
    and generates a Grad-CAM overlay backpropagated from the ACL risk score
    (`PipelineRunner.explain(..., vqc=acl_model)`) — not embedding energy —
    so the heatmap reflects what actually drove that prediction.
    """
    from qknee.xai.gradcam import overlay_heatmap

    t0 = time.perf_counter()
    batch = runner.ingest(slice_2d)
    features = runner.extract_resnet_features(batch)
    quantum_angles = runner.reduce_to_quantum_angles(features)
    t1 = time.perf_counter()

    acl_score = runner.classify(quantum_angles, vqc=acl_model)
    t2 = time.perf_counter()
    meniscus_score = runner.classify(quantum_angles, vqc=meniscus_model)
    t3 = time.perf_counter()

    gradcam_overlay: Optional[np.ndarray] = None
    try:
        heatmap = runner.explain(batch[:, 0], vqc=acl_model)
        gradcam_overlay = overlay_heatmap(heatmap, slice_2d)
    except Exception as exc:  # noqa: BLE001 - a failed heatmap shouldn't hide the risk scores
        logger.warning("Grad-CAM generation failed; showing risk scores without an overlay: %s", exc)

    # ingest + ResNet18 + PCA time is attributed to "resnet_latency_ms"; the
    # two quantum head evaluations are split into "quantum_latency_ms".
    feature_ms = (t1 - t0) * 1000
    quantum_ms = (t3 - t1) * 1000

    return InferenceResult(
        acl_risk=acl_score,
        meniscus_risk=meniscus_score,
        resnet_latency_ms=feature_ms,
        pca_latency_ms=0.0,  # folded into feature_ms above
        quantum_latency_ms=quantum_ms,
        total_latency_ms=feature_ms + quantum_ms,
        backend="live",
        gradcam_overlay=gradcam_overlay,
    )


# --------------------------------------------------------------------------- #
# Volume ingestion + tri-planar slicing
# --------------------------------------------------------------------------- #

def load_volume(uploaded_files) -> Optional[np.ndarray]:
    """Loads uploaded file(s) into a 3D (D, H, W) array via
    `DataIngestion.load_volume_array` — the same DICOM-series/.nii/
    .nii.gz/.npy/.dcm loading path used elsewhere in the pipeline, so the
    dashboard's tri-planar Axial/Coronal/Sagittal slicing (`get_slice`)
    works consistently over any of those formats.

    Args:
        uploaded_files: A single Streamlit `UploadedFile` (`.npy`,
            `.dcm`/`.dicom`, or `.nii`/`.nii.gz`), or a list of
            `UploadedFile`s — a multi-file DICOM series upload, stacked
            into one volume ordered by InstanceNumber/SliceLocation.

    Returns None (and shows a Streamlit error) if the file(s) can't be parsed.
    """
    from qknee.data.ingestion import DataIngestion, IngestionError

    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    source = files if len(files) > 1 else files[0]
    display_name = f"{len(files)}-file DICOM series" if len(files) > 1 else files[0].name

    try:
        volume = DataIngestion().load_volume_array(source)
    except IngestionError as exc:
        st.error(f"Failed to read '{display_name}': {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read '{display_name}': {exc}")
        return None

    volume = np.asarray(volume)
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]  # single slice -> (1, H, W)
    elif volume.ndim != 3:
        st.error(f"Expected a 2D slice or 3D volume, got array of shape {volume.shape}.")
        return None

    return volume


def get_slice(volume: np.ndarray, view: str, index: int) -> np.ndarray:
    """Extracts a 2D slice from a 3D (D, H, W) volume along the requested
    anatomical plane.

    - "Axial"    : index along axis 0 (D)  -> (H, W)
    - "Coronal"  : index along axis 1 (H)  -> (D, W)
    - "Sagittal" : index along axis 2 (W)  -> (D, H)
    """
    if view == "Axial":
        return volume[index, :, :]
    elif view == "Coronal":
        return volume[:, index, :]
    elif view == "Sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unknown view '{view}'")


def view_axis_size(volume: np.ndarray, view: str) -> int:
    axis = {"Axial": 0, "Coronal": 1, "Sagittal": 2}[view]
    return volume.shape[axis]


def normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = slice_2d.astype(np.float32)
    min_val, max_val = float(slice_2d.min()), float(slice_2d.max())
    if max_val > min_val:
        slice_2d = (slice_2d - min_val) / (max_val - min_val)
    else:
        slice_2d = np.zeros_like(slice_2d)
    return (slice_2d * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# UI components
# --------------------------------------------------------------------------- #

def render_header() -> None:
    st.set_page_config(
        page_title="Q-Knee Diagnostic Dashboard",
        page_icon="🦵",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .qknee-banner {
            padding: 0.9rem 1.2rem;
            border-radius: 0.6rem;
            background: linear-gradient(90deg, #101820 0%, #16222A 100%);
            border: 1px solid #22303C;
            margin-bottom: 1rem;
        }
        .qknee-disclaimer {
            font-size: 0.78rem;
            color: #8B949E;
            margin-top: -0.4rem;
        }
        </style>
        <div class="qknee-banner">
            <h2 style="margin-bottom:0;">🦵 Q-Knee Diagnostic Dashboard</h2>
            <span class="qknee-disclaimer">
                Research prototype — quantum-assisted ACL / meniscal tear risk triage.
                Not a certified medical device. Not for clinical use.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quantum_status(mode: str, backend_ready: bool, api_url: Optional[str]) -> None:
    st.sidebar.markdown("### Quantum Backend Status")
    if mode == "api":
        st.sidebar.markdown(f"🔵 **HTTP API Mode** — querying `{api_url}/predict`")
        st.sidebar.caption("Inference runs in the API container; per-stage latency is not reported this way.")
    elif backend_ready:
        st.sidebar.markdown(
            "🟢 **NISQ Simulator Active** — PennyLane `default.qubit`, 4 qubits"
        )
        st.sidebar.caption("Angle-encoded VQC · variational depth = 3 layers")
    else:
        st.sidebar.markdown("🟡 **Mock Mode** — quantum backend unavailable")
        backend_error = st.session_state.get("_backend_error")
        if backend_error:
            st.sidebar.caption(f"Reason: {backend_error}")
        else:
            st.sidebar.caption("No fitted PCA artifact found (pca_scaler.pkl)")

    if api_url and mode != "api":
        st.sidebar.caption(f"$QKNEE_API_URL is set ({api_url}) but unreachable — using in-process/mock inference.")


def render_risk_gauge(label: str, value: Optional[float]) -> None:
    if value is None:
        st.metric(label=f"⚪ {label} Tear Risk", value="N/A", delta="unavailable in API mode")
        st.progress(0.0)
        return

    if value >= 0.66:
        color, tier = "🔴", "HIGH"
    elif value >= 0.33:
        color, tier = "🟠", "MODERATE"
    else:
        color, tier = "🟢", "LOW"

    st.metric(label=f"{color} {label} Tear Risk", value=f"{value * 100:.1f}%", delta=tier)
    st.progress(min(max(value, 0.0), 1.0))


def render_gradcam_panel(result: InferenceResult) -> None:
    is_api = result.backend.startswith("api")
    st.markdown(f"#### Grad-CAM ({'unified' if is_api else 'ACL'} risk)")
    if result.gradcam_overlay is not None:
        caption = (
            "Regions driving the predicted risk score"
            if is_api else
            "Regions driving the ACL tear-risk prediction"
        )
        st.image(result.gradcam_overlay, channels="BGR", use_container_width=True, caption=caption)
    else:
        st.info("Grad-CAM overlay unavailable for this slice.")


def render_latency_metrics(result: InferenceResult) -> None:
    st.markdown("#### Processing Latency")
    if result.backend.startswith("api"):
        st.metric("HTTP Round-Trip", f"{result.total_latency_ms:.1f} ms")
        st.caption(f"Per-stage timing isn't reported over HTTP ({result.backend} backend).")
        return

    cols = st.columns(3)
    cols[0].metric("Feature Extraction", f"{result.resnet_latency_ms:.1f} ms")
    cols[1].metric("PCA Reduction", f"{result.pca_latency_ms:.1f} ms")
    cols[2].metric("Quantum Circuit", f"{result.quantum_latency_ms:.1f} ms")
    st.caption(f"Total end-to-end latency: **{result.total_latency_ms:.1f} ms** "
               f"({result.backend} backend)")


def render_report_download(display_slice: np.ndarray, result: InferenceResult) -> None:
    """Renders a 'Download PDF Report' button compiling the current slice,
    Grad-CAM overlay, and ACL/meniscus risk scores into a radiology-style
    PDF via `qknee.xai.report_generator`. A failed generation degrades to
    a warning rather than crashing the dashboard."""
    from qknee.xai.report_generator import PatientMetadata, generate_radiology_report_bytes

    st.markdown("#### Report")
    try:
        pdf_bytes = generate_radiology_report_bytes(
            mri_slice=display_slice,
            acl_risk=result.acl_risk,
            meniscus_risk=result.meniscus_risk,
            gradcam_overlay=result.gradcam_overlay,
            patient_metadata=PatientMetadata(scan_description="Knee MRI — Q-Knee dashboard session"),
            backend=result.backend,
        )
    except Exception as exc:  # noqa: BLE001 - a failed report shouldn't crash the dashboard
        logger.warning("PDF report generation failed: %s", exc)
        st.warning("Could not generate the PDF report for this slice.")
        return

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name=f"qknee_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #

def main() -> None:
    render_header()

    pipeline, acl_model, meniscus_model = load_backend()
    backend_ready = pipeline is not None

    api_url = resolve_api_url()
    use_api = bool(api_url) and api_is_reachable(api_url)
    mode = "api" if use_api else ("live" if backend_ready else "mock")
    render_quantum_status(mode, backend_ready, api_url)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload MRI Volume")
    uploaded_files = st.sidebar.file_uploader(
        "DICOM series (.dcm, select all files), single DICOM, NumPy volume (.npy), or NIfTI (.nii/.nii.gz)",
        type=["dcm", "dicom", "npy", "nii", "gz"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload a `.npy`/`.nii`/`.nii.gz` volume or one or more `.dcm` files "
                 "(a full series) from the sidebar to begin, or explore the dashboard "
                 "below with synthetic demo data.")
        rng = np.random.default_rng(123)
        volume = rng.normal(loc=128, scale=40, size=(24, 224, 224)).clip(0, 255)
        st.sidebar.caption("Currently viewing: synthetic demo volume (24 slices)")
    else:
        volume = load_volume(uploaded_files)
        if volume is None:
            st.stop()
        display_name = (
            f"{len(uploaded_files)}-file DICOM series" if len(uploaded_files) > 1 else uploaded_files[0].name
        )
        st.sidebar.success(f"Loaded '{display_name}' — shape {volume.shape}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### View Controls")
    view = st.sidebar.radio(
        "Anatomical Plane", ["Axial", "Coronal", "Sagittal"], horizontal=False,
        help="Scroll slice-by-slice through the loaded 3D volume in any of the three anatomical planes.",
    )

    max_index = max(view_axis_size(volume, view) - 1, 0)
    slice_index = st.sidebar.slider("Slice", min_value=0, max_value=max_index, value=max_index // 2)

    raw_slice = get_slice(volume, view, slice_index)
    display_slice = normalize_for_display(raw_slice)

    result: Optional[InferenceResult] = None
    if use_api:
        try:
            result = run_api_inference(raw_slice, api_url)
        except Exception as exc:  # noqa: BLE001 - a failed request degrades to in-process/mock, not a crash
            logger.warning("API inference failed (%s); falling back to in-process/mock.", exc)

    if result is None:
        if backend_ready:
            result = run_live_inference(raw_slice, pipeline, acl_model, meniscus_model)
        else:
            result = run_mock_inference(raw_slice)

    image_col, gradcam_col, results_col = st.columns([1, 1, 1])

    with image_col:
        st.markdown(f"#### {view} View — Slice {slice_index}/{max_index}")
        st.image(display_slice, use_container_width=True, clamp=True)

    with gradcam_col:
        render_gradcam_panel(result)

    with results_col:
        st.markdown("#### Tear Risk Assessment")
        render_risk_gauge("ACL", result.acl_risk)
        render_risk_gauge("Meniscus", result.meniscus_risk)
        st.markdown("---")
        render_latency_metrics(result)

    st.markdown("---")
    render_report_download(display_slice, result)

    st.markdown("---")
    st.caption(
        "Pipeline: DICOM/NPY ingestion → torchvision preprocessing → frozen ResNet18 "
        "(512-D) → StandardScaler + PCA(4) → [0, 2π] angle scaling → 4-qubit PennyLane "
        "VQC → classical readout. Grad-CAM is backpropagated from the ACL risk score "
        "itself (not embedding energy), so the heatmap explains that prediction."
    )


if __name__ == "__main__":
    main()
