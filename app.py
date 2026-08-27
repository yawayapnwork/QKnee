"""
Q-Knee: interactive clinical diagnostic dashboard (Streamlit).

Lets a clinician/researcher upload a DICOM or .npy MRI volume, scroll through
slices in any of the three anatomical planes, and see real-time ACL/meniscus
tear-risk scores produced by the ResNet18 -> PCA -> 4-qubit VQC pipeline
built in mri_dataset.py / resnet_feature_extractor.py / quantum_dim_reduction.py
/ pipeline.py / vqc_classifier.py.

If the trained backend (PCA artifact, PyTorch/PennyLane models) isn't
available in the current environment, the dashboard transparently falls
back to a seeded mock inference engine so the UI remains fully demoable.

Run with:
    streamlit run app.py

RESEARCH PROTOTYPE — not a certified medical device. Not for clinical use.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Backend wiring (real pipeline if available, seeded mock fallback otherwise)
# --------------------------------------------------------------------------- #

PCA_ARTIFACT_PATH = Path("pca_scaler.pkl")


@dataclass
class InferenceResult:
    acl_risk: float          # [0, 1]
    meniscus_risk: float     # [0, 1]
    resnet_latency_ms: float
    pca_latency_ms: float
    quantum_latency_ms: float
    total_latency_ms: float
    backend: str             # "live" or "mock"


@st.cache_resource(show_spinner=False)
def load_backend():
    """Attempts to load the real ResNet18 -> PCA -> VQC pipeline.

    Returns a tuple (pipeline, acl_model, meniscus_model) or (None, None, None)
    if any dependency (torch, pennylane, or the fitted PCA artifact) is
    unavailable — the caller falls back to mock inference in that case.
    """
    if not PCA_ARTIFACT_PATH.exists():
        return None, None, None

    try:
        import torch

        from pipeline import MRIQuantumPipeline
        from vqc_classifier import VQCClassifier

        quantum_pipeline = MRIQuantumPipeline(pca_artifact_path=PCA_ARTIFACT_PATH)

        # Two independent quantum heads: one scored for ACL tear risk, one
        # for meniscus tear risk. NOTE: randomly initialized here — swap in
        # `VQCClassifier` weights loaded from a trained checkpoint for
        # real predictions.
        torch.manual_seed(42)
        acl_model = VQCClassifier()
        acl_model.eval()
        torch.manual_seed(7)
        meniscus_model = VQCClassifier()
        meniscus_model.eval()

        return quantum_pipeline, acl_model, meniscus_model
    except Exception as exc:  # noqa: BLE001 - surface any backend failure as "unavailable"
        st.session_state.setdefault("_backend_error", str(exc))
        return None, None, None


def _seed_from_slice(slice_2d: np.ndarray) -> int:
    """Deterministic seed derived from slice content, so mock scores stay
    stable for the same slice/view rather than flickering on every rerun."""
    digest = hashlib.sha256(slice_2d.tobytes()).digest()
    return int.from_bytes(digest[:4], "big")


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
    )


def run_live_inference(slice_2d: np.ndarray, pipeline, acl_model, meniscus_model) -> InferenceResult:
    """Runs the real ResNet18 -> PCA -> VQC pipeline on one 2D slice,
    timing each stage."""
    import torch

    t0 = time.perf_counter()
    quantum_vector = pipeline.extract_quantum_features(slice_2d, as_tensor=True, verbose=False)
    t1 = time.perf_counter()

    with torch.no_grad():
        acl_score = acl_model(quantum_vector).item()
        t2 = time.perf_counter()
        meniscus_score = meniscus_model(quantum_vector).item()
        t3 = time.perf_counter()

    # extract_quantum_features already covers ResNet18 + PCA; we don't have
    # separate sub-timers without modifying pipeline.py, so attribute the
    # combined ingestion+ResNet+PCA time to "resnet_latency_ms" and split
    # the two quantum head evaluations into "quantum_latency_ms".
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
    )


# --------------------------------------------------------------------------- #
# Volume ingestion + tri-planar slicing
# --------------------------------------------------------------------------- #

def load_volume(uploaded_file) -> Optional[np.ndarray]:
    """Loads an uploaded .npy or DICOM (.dcm) file into a 3D (D, H, W) array.

    Returns None (and shows a Streamlit error) if the file can't be parsed.
    """
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if suffix == ".npy":
            volume = np.load(uploaded_file)
        elif suffix in (".dcm", ".dicom"):
            import pydicom

            dataset = pydicom.dcmread(uploaded_file)
            volume = dataset.pixel_array
        else:
            st.error(f"Unsupported file type '{suffix}'. Upload a .npy or .dcm file.")
            return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read '{uploaded_file.name}': {exc}")
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


def render_quantum_status(backend_ready: bool) -> None:
    st.sidebar.markdown("### Quantum Backend Status")
    if backend_ready:
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


def render_risk_gauge(label: str, value: float) -> None:
    if value >= 0.66:
        color, tier = "🔴", "HIGH"
    elif value >= 0.33:
        color, tier = "🟠", "MODERATE"
    else:
        color, tier = "🟢", "LOW"

    st.metric(label=f"{color} {label} Tear Risk", value=f"{value * 100:.1f}%", delta=tier)
    st.progress(min(max(value, 0.0), 1.0))


def render_latency_metrics(result: InferenceResult) -> None:
    st.markdown("#### Processing Latency")
    cols = st.columns(3)
    cols[0].metric("Feature Extraction", f"{result.resnet_latency_ms:.1f} ms")
    cols[1].metric("PCA Reduction", f"{result.pca_latency_ms:.1f} ms")
    cols[2].metric("Quantum Circuit", f"{result.quantum_latency_ms:.1f} ms")
    st.caption(f"Total end-to-end latency: **{result.total_latency_ms:.1f} ms** "
               f"({result.backend} backend)")


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #

def main() -> None:
    render_header()

    pipeline, acl_model, meniscus_model = load_backend()
    backend_ready = pipeline is not None
    render_quantum_status(backend_ready)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload MRI Volume")
    uploaded_file = st.sidebar.file_uploader(
        "DICOM (.dcm) or NumPy volume (.npy)",
        type=["dcm", "dicom", "npy"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        st.info("Upload a `.npy` volume or `.dcm` file from the sidebar to begin, "
                 "or explore the dashboard below with synthetic demo data.")
        rng = np.random.default_rng(123)
        volume = rng.normal(loc=128, scale=40, size=(24, 224, 224)).clip(0, 255)
        st.sidebar.caption("Currently viewing: synthetic demo volume (24 slices)")
    else:
        volume = load_volume(uploaded_file)
        if volume is None:
            st.stop()
        st.sidebar.success(f"Loaded '{uploaded_file.name}' — shape {volume.shape}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### View Controls")
    view = st.sidebar.radio("Anatomical Plane", ["Axial", "Coronal", "Sagittal"], horizontal=False)

    max_index = max(view_axis_size(volume, view) - 1, 0)
    slice_index = st.sidebar.slider("Slice", min_value=0, max_value=max_index, value=max_index // 2)

    raw_slice = get_slice(volume, view, slice_index)
    display_slice = normalize_for_display(raw_slice)

    image_col, results_col = st.columns([1.1, 1])

    with image_col:
        st.markdown(f"#### {view} View — Slice {slice_index}/{max_index}")
        st.image(display_slice, use_container_width=True, clamp=True)

    with results_col:
        st.markdown("#### Tear Risk Assessment")

        if backend_ready:
            result = run_live_inference(raw_slice, pipeline, acl_model, meniscus_model)
        else:
            result = run_mock_inference(raw_slice)

        render_risk_gauge("ACL", result.acl_risk)
        render_risk_gauge("Meniscus", result.meniscus_risk)
        st.markdown("---")
        render_latency_metrics(result)

    st.markdown("---")
    st.caption(
        "Pipeline: DICOM/NPY ingestion → torchvision preprocessing → frozen ResNet18 "
        "(512-D) → StandardScaler + PCA(4) → [0, 2π] angle scaling → 4-qubit PennyLane "
        "VQC → classical readout."
    )


if __name__ == "__main__":
    main()
