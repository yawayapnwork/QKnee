"""
Q-Knee REST API: FastAPI wrapper exposing `qknee.models.pipeline.PipelineRunner`
(DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM) to the Streamlit frontend
(qknee/ui) or any other HTTP client.

Run with:
    uvicorn qknee.api.server:app --reload --port 8000

Endpoints:
    GET  /health   - liveness/readiness probe, reports whether the real
                     model backend loaded or the API is running in mock mode.
    POST /predict  - accepts a DICOM (.dcm/.dicom) or NumPy (.npy) MRI
                     slice/volume upload, returns risk score, diagnosis, and
                     a base64-encoded Grad-CAM heatmap overlay (PNG).
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger, setup_logging
from qknee.models.pipeline import PipelineRunner, PipelineValidationError
from qknee.xai.gradcam import overlay_heatmap

setup_logging()
logger = get_logger("qknee.api")

_config = load_config()
PCA_ARTIFACT_PATH = _config.paths.pca_artifact
TEAR_RISK_THRESHOLD = _config.api.tear_risk_threshold


# --------------------------------------------------------------------------- #
# Response schema
# --------------------------------------------------------------------------- #

class PredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Predicted tear risk probability, in [0, 1].")
    diagnosis: str = Field(..., description="'Tear Detected' if risk_score >= 0.5, else 'Normal'.")
    gradcam_heatmap: str = Field(..., description="Base64-encoded PNG of the Grad-CAM overlay on the input slice.")
    backend: str = Field(..., description="'live' if PipelineRunner ran, 'mock' if a fallback was used.")


class HealthResponse(BaseModel):
    status: str
    backend_ready: bool
    detail: Optional[str] = None


# --------------------------------------------------------------------------- #
# Backend loading (mirrors app.py's mock-fallback pattern, so the API stays
# usable for frontend development even without a fitted PCA artifact)
# --------------------------------------------------------------------------- #

class QKneeBackend:
    """Thin FastAPI-facing wrapper around `PipelineRunner` — the canonical
    DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM engine. Delegates all
    actual inference to one `PipelineRunner` instance built at startup, and
    falls back to a deterministic mock only when that engine fails to
    initialize (e.g. no fitted PCA artifact yet), so the API stays usable
    for frontend development without a trained backend."""

    def __init__(self, pca_artifact_path: Path = PCA_ARTIFACT_PATH):
        self.backend_ready = False
        self.load_error: Optional[str] = None
        self.runner: Optional[PipelineRunner] = None

        try:
            self.runner = PipelineRunner(config=_config, pca_artifact_path=pca_artifact_path)
            self.backend_ready = True
            logger.info("QKneeBackend loaded live PipelineRunner from %s", pca_artifact_path)
        except PipelineValidationError as exc:
            self.load_error = str(exc)
            logger.warning("QKneeBackend starting in MOCK mode: %s", self.load_error)
        except Exception as exc:  # noqa: BLE001
            self.load_error = str(exc)
            logger.warning("QKneeBackend starting in MOCK mode: %s", self.load_error)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_dicom_slice(raw_bytes: bytes) -> np.ndarray:
        """Parses DICOM bytes into a calibrated grayscale pixel array.

        Beyond a bare `pixel_array` read, this:
            - Reads with `force=True` so anonymized/non-conformant exports
              missing the 128-byte preamble + 'DICM' magic (common from some
              clinical PACS/de-identification pipelines) still parse instead
              of raising `InvalidDicomError`.
            - Applies the Modality LUT (`RescaleSlope`/`RescaleIntercept`, or
              a full `ModalityLUTSequence` if present) via pydicom, so CT-style
              DICOMs return calibrated intensities (e.g. Hounsfield units)
              instead of raw stored pixel values. A no-op when neither is
              present in the dataset.
            - Inverts `MONOCHROME1` datasets (where 0 = white, max = black)
              so intensity semantics match the far more common `MONOCHROME2`
              convention (0 = black) that the rest of the pipeline assumes.
        """
        import io

        import pydicom
        from pydicom.pixels import apply_modality_lut

        try:
            dataset = pydicom.dcmread(io.BytesIO(raw_bytes), force=True)
            array = apply_modality_lut(dataset.pixel_array, dataset)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Failed to parse DICOM file: {exc}") from exc

        if getattr(dataset, "PhotometricInterpretation", None) == "MONOCHROME1":
            array = np.asarray(array)
            array = array.max() - array
            logger.debug("Inverted MONOCHROME1 DICOM slice to MONOCHROME2 intensity convention.")

        return array

    def load_slice(self, raw_bytes: bytes, filename: str) -> np.ndarray:
        """Parses uploaded bytes (.dcm/.dicom or .npy) into a single 2D
        grayscale slice array. Multi-slice .npy volumes use their middle slice."""
        suffix = Path(filename).suffix.lower()

        if suffix in (".dcm", ".dicom"):
            array = self._load_dicom_slice(raw_bytes)

        elif suffix == ".npy":
            import io

            try:
                array = np.load(io.BytesIO(raw_bytes))
            except Exception as exc:
                raise HTTPException(status_code=422, detail=f"Failed to parse .npy file: {exc}") from exc

        else:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix}'. Upload a .dcm/.dicom or .npy file.",
            )

        array = np.asarray(array)
        if array.ndim == 3:
            array = array[array.shape[0] // 2]  # middle slice of a volume
        elif array.ndim != 2:
            raise HTTPException(
                status_code=422,
                detail=f"Expected a 2D slice or 3D volume, got array of shape {array.shape}.",
            )

        return array

    @staticmethod
    def _normalize_uint8(slice_2d: np.ndarray) -> np.ndarray:
        slice_2d = slice_2d.astype(np.float32)
        min_val, max_val = float(slice_2d.min()), float(slice_2d.max())
        if max_val > min_val:
            slice_2d = (slice_2d - min_val) / (max_val - min_val)
        else:
            slice_2d = np.zeros_like(slice_2d)
        return (slice_2d * 255).astype(np.uint8)

    def predict(self, raw_bytes: bytes, filename: str) -> PredictionResponse:
        slice_2d = self.load_slice(raw_bytes, filename)
        display_slice = self._normalize_uint8(slice_2d)

        if self.backend_ready:
            risk_score, heatmap_b64 = self._predict_live(display_slice)
            backend = "live"
        else:
            risk_score, heatmap_b64 = self._predict_mock(display_slice)
            backend = "mock"

        diagnosis = "Tear Detected" if risk_score >= TEAR_RISK_THRESHOLD else "Normal"

        return PredictionResponse(
            risk_score=risk_score,
            diagnosis=diagnosis,
            gradcam_heatmap=heatmap_b64,
            backend=backend,
        )

    def _predict_live(self, display_slice: np.ndarray) -> Tuple[float, str]:
        try:
            result = self.runner.run(display_slice)  # DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM
            overlay = overlay_heatmap(result.gradcam_heatmap, display_slice)
            heatmap_b64 = self._encode_png_base64(overlay)
            return result.risk_score, heatmap_b64
        except Exception as exc:  # noqa: BLE001 - PipelineValidationError or any unexpected failure
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    def _predict_mock(self, display_slice: np.ndarray) -> Tuple[float, str]:
        import hashlib

        digest = hashlib.sha256(display_slice.tobytes()).digest()
        seed = int.from_bytes(digest[:4], "big")
        rng = np.random.default_rng(seed)
        risk_score = float(rng.uniform(0.05, 0.95))

        # Fake "heatmap": a soft radial gradient, just so the response shape
        # (and the frontend's image-decoding path) is exercised in mock mode.
        h, w = display_slice.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        fake_heatmap = np.clip(1 - radius / radius.max(), 0, 1)
        overlay = overlay_heatmap(fake_heatmap, display_slice)
        return risk_score, self._encode_png_base64(overlay)

    @staticmethod
    def _encode_png_base64(bgr_image: np.ndarray) -> str:
        success, encoded = cv2.imencode(".png", bgr_image)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode Grad-CAM heatmap as PNG")
        return base64.b64encode(encoded.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Q-Knee Inference API",
    description="Quantum-assisted ACL/meniscal tear risk scoring from MRI slices.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

backend = QKneeBackend()


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health() -> HealthResponse:
    """Reports whether the live QKneeModel backend loaded successfully."""
    return HealthResponse(
        status="ok",
        backend_ready=backend.backend_ready,
        detail=backend.load_error,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="DICOM (.dcm/.dicom) or NumPy (.npy) MRI slice/volume")) -> PredictionResponse:
    """Runs one MRI slice (or the middle slice of a volume) through the
    Q-Knee pipeline and returns the tear-risk score, diagnosis label, and a
    base64-encoded Grad-CAM overlay for visual explainability."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    return backend.predict(raw_bytes, file.filename or "upload")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("qknee.api.server:app", host=_config.api.host, port=_config.api.port, reload=True)
