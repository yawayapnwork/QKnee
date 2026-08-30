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
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Demo Mode / Latency Fallback: precomputed predictions + Grad-CAM heatmaps
# for a handful of sample cases, built offline by `scripts/build_demo_cache.py`,
# so live judging never waits on a cold model/QNode.
DEMO_CACHE_DIR = Path("qknee/artifacts/demo_cache")
DEMO_CACHE_INDEX_PATH = DEMO_CACHE_DIR / "index.json"

# Quantum-vs-classical comparative benchmark, built offline by
# `scripts/run_benchmark.py`.
BENCHMARK_RESULTS_PATH = Path("qknee/artifacts/benchmark_results.json")
BENCHMARK_ROC_PATH = Path("qknee/artifacts/benchmark_roc_curve.png")

# The PRD's Plan B latency-risk mitigation cache (`scripts/generate_demo_cache.py`)
# — 10 cases' full pipeline outputs (quantum angles, Pauli-Z expectations,
# risk scores, Grad-CAM overlays) serialized to one JSON file, with each
# heatmap also embedded as base64 so nothing needs a second filesystem
# read. Pre-warmed into `@st.cache_resource` memory at startup (see
# `load_precomputed_cache` and its eager call in `main()`) — a cold-start
# handler distinct from (and in addition to) `load_demo_cache_index`'s
# sidebar-toggle cache above.
PRECOMPUTED_CACHE_PATH = Path("qknee/artifacts/precomputed_cache.json")


@dataclass
class InferenceResult:
    acl_risk: float                    # [0, 1]
    meniscus_risk: Optional[float]     # [0, 1], or None when unavailable (e.g. "api" backend)
    resnet_latency_ms: float
    pca_latency_ms: float
    quantum_latency_ms: float
    total_latency_ms: float
    backend: str             # "live", "mock", "api", or "cached/..."
    gradcam_overlay: Optional[np.ndarray] = None  # (H, W, 3) BGR uint8 pre-blended overlay, or None
    gradcam_heatmap: Optional[np.ndarray] = None  # (h, w) float32 in [0, 1], raw Grad-CAM — lets the
    # UI re-blend at any opacity live (see `render_gradcam_panel`) without re-running inference.
    # None for backends that only ever produce a pre-blended overlay (HTTP API mode).


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


def _mock_gradcam_heatmap(slice_2d: np.ndarray) -> np.ndarray:
    """Cheap, torch-free stand-in *raw* Grad-CAM heatmap for mock mode (a
    soft radial gradient), so the heatmap panel — including its live
    opacity slider — is exercised in the UI even without a live backend.
    Blended into a displayable overlay via `qknee.xai.gradcam.overlay_heatmap`,
    the same function the live backend uses, so mock/live share one blending
    implementation."""
    display = normalize_for_display(slice_2d)
    height, width = display.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = height / 2, width / 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return np.clip(1 - radius / radius.max(), 0, 1).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=128)
def run_mock_inference(slice_2d: np.ndarray) -> InferenceResult:
    """Seeded, deterministic mock scores + plausible latency numbers, used
    whenever the real pipeline/model backend isn't available.

    Cached by `slice_2d` content: Streamlit reruns this whole script on
    every widget interaction (e.g. dragging the slice slider back to a
    slice already viewed), so without this cache the same slice's mock
    Grad-CAM heatmap would be regenerated from scratch on every one of
    those reruns even though the result is already deterministic."""
    from qknee.xai.gradcam import overlay_heatmap

    rng = np.random.default_rng(_seed_from_slice(slice_2d))

    resnet_ms = float(rng.uniform(18, 35))
    pca_ms = float(rng.uniform(0.5, 2.0))
    quantum_ms = float(rng.uniform(4, 12))

    heatmap = _mock_gradcam_heatmap(slice_2d)

    return InferenceResult(
        acl_risk=float(rng.uniform(0.05, 0.95)),
        meniscus_risk=float(rng.uniform(0.05, 0.95)),
        resnet_latency_ms=resnet_ms,
        pca_latency_ms=pca_ms,
        quantum_latency_ms=quantum_ms,
        total_latency_ms=resnet_ms + pca_ms + quantum_ms,
        backend="mock",
        gradcam_overlay=overlay_heatmap(heatmap, slice_2d),
        gradcam_heatmap=heatmap,
    )


@st.cache_data(show_spinner=False, max_entries=128)
def run_live_inference(slice_2d: np.ndarray, _runner, _acl_model, _meniscus_model) -> InferenceResult:
    """Runs the real DataIngestion -> ResNet18 -> PCA -> VQC pipeline (via
    `PipelineRunner`'s stage methods) on one 2D slice, timing each stage,
    and generates a Grad-CAM overlay backpropagated from the ACL risk score
    (`PipelineRunner.explain(..., vqc=acl_model)`) — not embedding energy —
    so the heatmap reflects what actually drove that prediction.

    Cached by `slice_2d` content only: the leading underscore on
    `_runner`/`_acl_model`/`_meniscus_model` tells `st.cache_data` to
    exclude them from the cache key (they're not hashable torch/PennyLane
    objects, and are effectively constant for the app process's lifetime
    anyway). Without this, the full ResNet18 forward pass and Grad-CAM
    backprop — the expensive part of every rerun — would re-execute on
    every Streamlit script rerun, including ones triggered by a completely
    unrelated widget, as long as the viewed slice hasn't actually changed.
    """
    from qknee.xai.gradcam import overlay_heatmap

    runner, acl_model, meniscus_model = _runner, _acl_model, _meniscus_model

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
    gradcam_heatmap: Optional[np.ndarray] = None
    try:
        gradcam_heatmap = runner.explain(batch[:, 0], vqc=acl_model)
        gradcam_overlay = overlay_heatmap(gradcam_heatmap, slice_2d)
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
        gradcam_heatmap=gradcam_heatmap,
    )


# --------------------------------------------------------------------------- #
# Volume ingestion + tri-planar slicing
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Decoding uploaded MRI volume...", max_entries=8)
def _decode_volume_cached(file_payloads: Tuple[Tuple[str, bytes], ...]) -> np.ndarray:
    """Pure DICOM-series/.nii/.nii.gz/.npy volume decode — the actual
    expensive step `load_volume` wraps. Keyed on `(filename, bytes)` tuples
    (real file content) rather than Streamlit `UploadedFile` objects, and
    deliberately free of any `st.*` calls: Streamlit only replays a cached
    function's *return value* on a cache hit, not side effects like
    `st.error` performed inside it, so those must live in the uncached
    `load_volume` wrapper below instead.

    Without this cache, the same uploaded DICOM series/NIfTI/.npy volume
    would be fully re-decoded on every Streamlit script rerun — which
    includes every slice-slider drag and every unrelated widget
    interaction, not just a new upload.
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


def load_volume(uploaded_files) -> Optional[np.ndarray]:
    """Loads uploaded file(s) into a 3D (D, H, W) array via
    `DataIngestion.load_volume_array` (through the cached
    `_decode_volume_cached` above) — the same DICOM-series/.nii/
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
    from qknee.data.ingestion import IngestionError

    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    display_name = f"{len(files)}-file DICOM series" if len(files) > 1 else files[0].name
    file_payloads = tuple((f.name, f.getvalue()) for f in files)

    try:
        volume = _decode_volume_cached(file_payloads)
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


def render_gradcam_panel(display_slice: np.ndarray, result: InferenceResult) -> None:
    """Renders the Grad-CAM panel. When the raw heatmap is available
    (`result.gradcam_heatmap` — live/mock/cached backends), an opacity
    slider re-blends it onto `display_slice` live via
    `qknee.xai.gradcam.overlay_heatmap` on every drag — no re-inference,
    just a resize + colormap + `cv2.addWeighted`, so it's effectively
    free. HTTP API mode only ever returns a pre-blended overlay, so the
    slider is hidden there and the static image is shown instead."""
    is_api = result.backend.startswith("api")
    st.markdown(f"#### Grad-CAM ({'unified' if is_api else 'ACL'} risk)")

    caption = (
        "Regions driving the predicted risk score"
        if is_api else
        "Regions driving the ACL tear-risk prediction"
    )

    if result.gradcam_heatmap is not None:
        from qknee.xai.gradcam import overlay_heatmap

        opacity = st.slider(
            "Heatmap Opacity", min_value=0.0, max_value=1.0,
            value=float(_config.gradcam.alpha), step=0.05,
            key=f"gradcam_opacity_{result.backend}",
            help="Blends the raw Grad-CAM heatmap onto the slice live — no re-inference needed.",
        )
        overlay = overlay_heatmap(result.gradcam_heatmap, display_slice, alpha=opacity)
        st.image(overlay, channels="BGR", use_container_width=True, caption=caption)
    elif result.gradcam_overlay is not None:
        st.image(result.gradcam_overlay, channels="BGR", use_container_width=True, caption=caption)
        st.caption("Opacity control unavailable — this backend only returns a pre-rendered overlay.")
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


# --------------------------------------------------------------------------- #
# Demo Mode / Latency Fallback: precomputed "NISQ cache" of sample cases
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def load_precomputed_cache() -> Optional[Dict]:
    """Cold-start pre-warm hook: eagerly parses
    `qknee/artifacts/precomputed_cache.json` into `@st.cache_resource`
    process memory (see the eager call in `main()`) so its JSON-parse/disk-I/O
    cost is paid once at boot, before any user's first request, rather than
    lazily on whichever request happens to touch it first.

    `cache_resource` (process-scoped, shared read-only across every
    session on this container) rather than `cache_data` (which would
    additionally pickle/copy the payload per session) — same reasoning as
    `load_backend()`'s model objects above.

    Returns `None` (logged) if the cache hasn't been built yet
    (`python scripts/generate_demo_cache.py`) or fails to parse.
    """
    if not PRECOMPUTED_CACHE_PATH.exists():
        logger.info(
            "No precomputed cache found at %s; skipping cold-start pre-warm "
            "(run `python scripts/generate_demo_cache.py` to build one).",
            PRECOMPUTED_CACHE_PATH,
        )
        return None
    try:
        payload = json.loads(PRECOMPUTED_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read precomputed cache at %s: %s", PRECOMPUTED_CACHE_PATH, exc)
        return None
    logger.info(
        "Pre-warmed precomputed cache: %d case(s) from %s", payload.get("n_cases", 0), PRECOMPUTED_CACHE_PATH,
    )
    return payload


@st.cache_data(show_spinner=False)
def load_demo_cache_index() -> Optional[List[Dict]]:
    """Loads `qknee/artifacts/demo_cache/index.json` (built offline by
    `scripts/build_demo_cache.py`) — a handful of MRNet-style sample cases
    with precomputed risk scores, latency figures, and raw Grad-CAM
    heatmaps. Returns `None` if no cache has been built yet, so the sidebar
    toggle can disable itself with a clear message instead of erroring."""
    if not DEMO_CACHE_INDEX_PATH.exists():
        return None
    try:
        payload = json.loads(DEMO_CACHE_INDEX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read demo cache index at %s: %s", DEMO_CACHE_INDEX_PATH, exc)
        return None
    return payload.get("cases", [])


def load_cached_case(case: Dict) -> Tuple[np.ndarray, InferenceResult]:
    """Loads one demo-cache case's slice image + raw Grad-CAM heatmap +
    precomputed scores/latency straight from disk — no model, no PennyLane
    QNode, no ResNet forward pass. This is what makes "Use Precomputed NISQ
    Cache" genuinely zero-latency rather than just "fast": nothing in this
    function does any inference at all."""
    import cv2

    slice_path = DEMO_CACHE_DIR / case["slice_file"]
    display_slice = cv2.imread(str(slice_path), cv2.IMREAD_GRAYSCALE)
    if display_slice is None:
        raise FileNotFoundError(f"Demo cache slice image missing/unreadable: {slice_path}")

    heatmap: Optional[np.ndarray] = None
    if case.get("heatmap_file"):
        heatmap_path = DEMO_CACHE_DIR / case["heatmap_file"]
        if heatmap_path.exists():
            heatmap = np.load(heatmap_path)

    result = InferenceResult(
        acl_risk=float(case["acl_risk"]),
        meniscus_risk=(float(case["meniscus_risk"]) if case.get("meniscus_risk") is not None else None),
        resnet_latency_ms=float(case.get("resnet_latency_ms", 0.0)),
        pca_latency_ms=float(case.get("pca_latency_ms", 0.0)),
        quantum_latency_ms=float(case.get("quantum_latency_ms", 0.0)),
        total_latency_ms=float(case.get("total_latency_ms", 0.0)),
        backend=f"cached/{case.get('backend', 'unknown')}",
        gradcam_heatmap=heatmap,
    )
    return display_slice, result


def render_demo_cache_sidebar() -> Tuple[bool, Optional[Dict]]:
    """Renders the 'Use Precomputed NISQ Cache' sidebar toggle plus (when
    enabled) a sample-case picker. Returns `(use_cache, selected_case)` —
    `selected_case` is `None` unless `use_cache` is True and at least one
    case is available."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Demo Mode")

    cache_index = load_demo_cache_index()
    cache_available = bool(cache_index)

    use_cache = st.sidebar.toggle(
        "Use Precomputed NISQ Cache",
        value=False,
        disabled=not cache_available,
        help="Instantly replays precomputed predictions + Grad-CAM heatmaps for a handful "
             "of sample MRNet cases — zero inference latency, for live judging.",
    )

    if not cache_available:
        st.sidebar.caption(
            "No demo cache found. Run `python scripts/build_demo_cache.py` to generate "
            f"one at `{DEMO_CACHE_DIR}`."
        )
        return False, None

    if not use_cache:
        st.sidebar.caption(f"{len(cache_index)} precomputed case(s) available.")
        return False, None

    case_labels = [f"Case {case['case_id']}" for case in cache_index]
    selected_label = st.sidebar.selectbox("Sample Case", case_labels)
    selected_case = cache_index[case_labels.index(selected_label)]
    st.sidebar.success(f"Loaded precomputed case '{selected_case['case_id']}' — 0 ms inference.")
    return True, selected_case


# --------------------------------------------------------------------------- #
# Quantum vs. Classical benchmark tab
# --------------------------------------------------------------------------- #

def render_benchmark_tab() -> None:
    """Renders the offline comparative benchmark (`scripts/run_benchmark.py`)
    — ROC-AUC comparison and inference-latency-per-sample side by side, plus
    a full metrics table (F1/Precision/Recall/Confusion Matrix are in the
    underlying JSON; the table below surfaces the headline numbers)."""
    st.markdown("### Quantum vs. Classical Benchmark")
    st.caption(
        "Offline comparison of three architectures on the same MRNet-style validation "
        "subset (see `scripts/run_benchmark.py`) — not re-run live. Re-run that script to refresh."
    )

    if not BENCHMARK_RESULTS_PATH.exists():
        st.info(
            f"No benchmark results found at `{BENCHMARK_RESULTS_PATH}`. Run "
            "`python scripts/run_benchmark.py` to generate them."
        )
        return

    try:
        payload = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Failed to read {BENCHMARK_RESULTS_PATH}: {exc}")
        return

    models = payload.get("models", [])
    if not models:
        st.info("Benchmark results file has no models recorded.")
        return

    dataset_info = payload.get("dataset", {})
    generated_at = payload.get("generated_at", "unknown")
    st.caption(
        f"Generated {generated_at} · dataset: {dataset_info.get('source', '?')} "
        f"({dataset_info.get('n_test', '?')} test samples, "
        f"{dataset_info.get('plane', '?')} plane)"
    )

    import pandas as pd

    roc_col, latency_col = st.columns(2)

    with roc_col:
        st.markdown("#### ROC-AUC Comparison")
        if BENCHMARK_ROC_PATH.exists():
            st.image(str(BENCHMARK_ROC_PATH), use_container_width=True)
        else:
            roc_df = pd.DataFrame({
                "Model": [m["name"] for m in models],
                "ROC-AUC": [m["roc_auc"] for m in models],
            }).set_index("Model")
            st.bar_chart(roc_df)
            st.caption(f"Pre-rendered ROC curve not found at `{BENCHMARK_ROC_PATH}`; showing a bar chart instead.")

    with latency_col:
        st.markdown("#### Inference Latency (ms/sample)")
        latency_df = pd.DataFrame({
            "Model": [m["name"] for m in models],
            "Latency (ms)": [m.get("latency_ms_per_sample") or 0.0 for m in models],
        }).set_index("Model")
        st.bar_chart(latency_df)
        st.caption("Single-sample (batch-size-1) wall-clock latency — reflects real one-slice-at-a-time inference.")

    st.markdown("#### Full Metrics")
    metrics_df = pd.DataFrame([
        {
            "Model": m["name"],
            "ROC-AUC": m["roc_auc"],
            "F1": m["f1_score"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Latency (ms/sample)": m.get("latency_ms_per_sample"),
            "Test Samples": m.get("n_test_samples"),
        }
        for m in models
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def render_report_download(display_slice: np.ndarray, result: InferenceResult) -> None:
    """Renders a 'Download Radiology PDF Report' button, in the summary
    (Tear Risk Assessment) panel, compiling the current slice, Grad-CAM
    overlay, and ACL/meniscus risk scores into a one-page radiology-style
    PDF via `qknee.xai.report_generator`. A failed generation degrades to
    a warning rather than crashing the dashboard."""
    from qknee.xai.gradcam import overlay_heatmap
    from qknee.xai.report_generator import generate_radiology_report

    # Cached (demo-mode) and mock/live results carry the raw heatmap rather
    # than a pre-blended overlay; build one at the default opacity for the
    # PDF if only the raw heatmap is available.
    gradcam_overlay = result.gradcam_overlay
    if gradcam_overlay is None and result.gradcam_heatmap is not None:
        gradcam_overlay = overlay_heatmap(result.gradcam_heatmap, display_slice)

    st.markdown("#### Report")
    try:
        pdf_bytes = generate_radiology_report(
            output_path=None,
            mri_slice=display_slice,
            gradcam_overlay=gradcam_overlay,
            prediction_results={
                "acl_risk": result.acl_risk,
                "meniscus_risk": result.meniscus_risk,
                "resnet_latency_ms": result.resnet_latency_ms,
                "pca_latency_ms": result.pca_latency_ms,
                "quantum_latency_ms": result.quantum_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "backend": result.backend,
            },
            metadata={
                "modality": "MRI Knee",
                "clinical_indication": "Q-Knee dashboard session",
                "scan_date": datetime.now().strftime("%Y-%m-%d"),
            },
        )
    except Exception as exc:  # noqa: BLE001 - a failed report shouldn't crash the dashboard
        logger.warning("PDF report generation failed: %s", exc)
        st.warning("Could not generate the PDF report for this slice.")
        return

    st.download_button(
        label="📄 Download Radiology PDF Report",
        data=pdf_bytes,
        file_name=f"qknee_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #

def render_diagnostic_tab() -> None:
    pipeline, acl_model, meniscus_model = load_backend()
    backend_ready = pipeline is not None

    api_url = resolve_api_url()
    use_api = bool(api_url) and api_is_reachable(api_url)
    mode = "api" if use_api else ("live" if backend_ready else "mock")
    render_quantum_status(mode, backend_ready, api_url)

    use_demo_cache, cached_case = render_demo_cache_sidebar()

    if use_demo_cache and cached_case is not None:
        # Demo Mode: skip upload/plane/slice/inference entirely and replay a
        # precomputed case straight from disk — genuinely zero-latency.
        display_slice, result = load_cached_case(cached_case)
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Upload MRI Volume")
        uploaded_files = st.sidebar.file_uploader(
            "DICOM series (.dcm, select all files — drag & drop supported), single DICOM, "
            "NumPy volume (.npy), or NIfTI (.nii/.nii.gz)",
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

        result = None
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
        if use_demo_cache and cached_case is not None:
            st.markdown(f"#### Cached Case {cached_case['case_id']} ({cached_case.get('plane', 'sagittal')} plane)")
        else:
            st.markdown(f"#### {view} View — Slice {slice_index}/{max_index}")
        st.image(display_slice, use_container_width=True, clamp=True)

    with gradcam_col:
        render_gradcam_panel(display_slice, result)

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


def main() -> None:
    # Pre-warmed cold-start handler: called eagerly, before any UI renders,
    # so the precomputed cache is already resident in `@st.cache_resource`
    # process memory by the time the first real user interaction happens —
    # `st.cache_resource` runs its body once per process and reuses the
    # result across every subsequent call/session, so this "first boot"
    # call is what actually does the pre-warming.
    precomputed_cache = load_precomputed_cache()

    render_header()
    if precomputed_cache is not None:
        st.sidebar.caption(f"⚡ Pre-warmed: {precomputed_cache.get('n_cases', 0)} precomputed case(s) resident in memory.")

    tab_diagnostic, tab_benchmark = st.tabs(["🔬 Diagnostic View", "📊 Quantum vs Classical Benchmark"])

    with tab_diagnostic:
        render_diagnostic_tab()

    with tab_benchmark:
        render_benchmark_tab()


if __name__ == "__main__":
    main()
