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

Falls back to a deterministic mock pipeline if the trained QKneeModel
backend (pca_scaler.pkl) isn't available, so the UI is fully demoable
without a live backend.

Run with:
    streamlit run qknee_frontend.py
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PCA_ARTIFACT_PATH = Path(os.environ.get("PCA_ARTIFACT_PATH", "pca_scaler.pkl"))
RISK_THRESHOLD = 0.5


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


# --------------------------------------------------------------------------- #
# Backend loading + inference (mock-fallback pattern shared with app.py)
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_backend():
    """Loads the real QKneeModel (ResNet18 -> PCA -> VQC) if a fitted PCA
    artifact is available; returns None otherwise so the UI can fall back
    to a deterministic mock."""
    if not PCA_ARTIFACT_PATH.exists():
        return None

    try:
        from model_pipeline import QKneeModel
        from quantum_dim_reduction import QuantumDimReducer

        reducer = QuantumDimReducer.load(PCA_ARTIFACT_PATH)
        model = QKneeModel(pca_reducer=reducer, n_qubits=4, n_layers=3)
        model.eval()
        return model
    except Exception as exc:  # noqa: BLE001
        st.session_state["_backend_error"] = str(exc)
        return None


def run_live_analysis(model, slice_2d: np.ndarray) -> AnalysisResult:
    import torch

    from mri_dataset import build_transforms
    from PIL import Image

    transform = build_transforms(train=False)
    pil_image = Image.fromarray(slice_2d, mode="L")
    input_tensor = transform(pil_image).unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        features = model.resnet(input_tensor)
        angles = model.pca_layer(features)
    t1 = time.perf_counter()
    with torch.no_grad():
        risk_score = float(model.vqc(angles).item())
    t2 = time.perf_counter()

    quantum_latency_ms = (t2 - t1) * 1000
    total_latency_ms = (t2 - t0) * 1000
    label = "Abnormality Detected" if risk_score >= RISK_THRESHOLD else "Normal"

    return AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=quantum_latency_ms,
        total_latency_ms=total_latency_ms,
        backend="live",
    )


def run_mock_analysis(slice_2d: np.ndarray) -> AnalysisResult:
    """Deterministic (seeded from slice content) mock, used when no trained
    backend is available."""
    digest = hashlib.sha256(slice_2d.tobytes()).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)

    quantum_latency_ms = float(rng.uniform(4, 12))
    total_latency_ms = quantum_latency_ms + float(rng.uniform(20, 35))
    risk_score = float(rng.uniform(0.05, 0.95))
    label = "Abnormality Detected" if risk_score >= RISK_THRESHOLD else "Normal"

    return AnalysisResult(
        risk_score=risk_score,
        prediction_label=label,
        quantum_latency_ms=quantum_latency_ms,
        total_latency_ms=total_latency_ms,
        backend="mock",
    )


def run_analysis(slice_2d: np.ndarray) -> AnalysisResult:
    model = load_backend()
    if model is not None:
        return run_live_analysis(model, slice_2d)
    return run_mock_analysis(slice_2d)


# --------------------------------------------------------------------------- #
# Image loading + display adjustment
# --------------------------------------------------------------------------- #

def load_scan(uploaded_file) -> Optional[np.ndarray]:
    """Loads a .png/.jpg/.jpeg (single slice) or .npy (single slice or
    volume) upload into a 3D (D, H, W) array."""
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if suffix == ".npy":
            array = np.load(uploaded_file)
        elif suffix in (".png", ".jpg", ".jpeg"):
            from PIL import Image

            array = np.array(Image.open(uploaded_file).convert("L"))
        else:
            st.error(f"Unsupported file type '{suffix}'. Upload a .png, .jpg, or .npy file.")
            return None
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read '{uploaded_file.name}': {exc}")
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


def render_sidebar() -> tuple[Optional[np.ndarray], int, float]:
    st.sidebar.markdown("### 📤 Upload Scan")
    uploaded_file = st.sidebar.file_uploader(
        "PNG, JPG, or NumPy volume (.npy)",
        type=["png", "jpg", "jpeg", "npy"],
    )

    backend = load_backend()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚛️ Backend Status")
    if backend is not None:
        st.sidebar.markdown("🟢 **Live model loaded** (ResNet18 → PCA → 4-qubit VQC)")
    else:
        st.sidebar.markdown("🟡 **Mock mode** — no trained backend found")
        error = st.session_state.get("_backend_error")
        if error:
            st.sidebar.caption(f"Reason: {error}")

    if uploaded_file is None:
        st.sidebar.info("Upload a scan to enable slice/contrast controls.")
        return None, 0, 1.0

    volume = load_scan(uploaded_file)
    if volume is None:
        return None, 0, 1.0

    st.sidebar.success(f"Loaded '{uploaded_file.name}' — {volume.shape[0]} slice(s)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎚️ View Controls")

    max_index = volume.shape[0] - 1
    if max_index > 0:
        slice_index = st.sidebar.slider("Slice", 0, max_index, max_index // 2)
    else:
        slice_index = 0
        st.sidebar.caption("Single-slice image — slider disabled.")

    contrast = st.sidebar.slider("Contrast", min_value=0.5, max_value=3.0, value=1.0, step=0.1)

    return volume, slice_index, contrast


def main() -> None:
    render_header()
    volume, slice_index, contrast = render_sidebar()

    if volume is None:
        st.info("👈 Upload a `.png`, `.jpg`, or `.npy` scan from the sidebar to begin.")
        return

    raw_slice = volume[slice_index]
    display_slice = apply_contrast(normalize_uint8(raw_slice), contrast)

    image_col, action_col = st.columns([1.3, 1])

    with image_col:
        st.markdown(f"#### Slice {slice_index + 1} / {volume.shape[0]}")
        st.image(display_slice, use_container_width=True, clamp=True)

    with action_col:
        st.markdown("#### Run Analysis")
        st.caption("Runs ResNet18 feature extraction + the 4-qubit quantum classifier on the slice above.")
        run_clicked = st.button("🔬 Run Q-Knee Analysis", type="primary", use_container_width=True)

        result_key = "last_analysis_result"
        if run_clicked:
            with st.spinner("Extracting features and running the quantum circuit..."):
                st.session_state[result_key] = run_analysis(raw_slice)

        result: Optional[AnalysisResult] = st.session_state.get(result_key)

        if result is None:
            st.info("Click **Run Q-Knee Analysis** to generate a prediction for this slice.")
        else:
            render_prediction_badge(result)

            gauge_col, latency_col = st.columns([1, 1])
            with gauge_col:
                st.pyplot(render_risk_gauge(result.risk_score), use_container_width=True)
            with latency_col:
                st.metric("Quantum Circuit Latency", f"{result.quantum_latency_ms:.1f} ms")
                st.metric("Total Pipeline Latency", f"{result.total_latency_ms:.1f} ms")
                st.caption(f"Backend: **{result.backend}**")

    st.markdown("---")
    st.caption(
        "Pipeline: upload → contrast-adjusted preview → ResNet18 (512-D) → "
        "PCA → [0, 2π] angle scaling → 4-qubit PennyLane VQC → risk score."
    )


if __name__ == "__main__":
    main()
