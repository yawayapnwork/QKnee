"""
Tests for `qknee.api.server` (the FastAPI backend) using FastAPI's
`TestClient`. Covers:

    1. `/health`, `/predict`, and `/explain` payload shape/contents in both
       live and mock backend modes.
    2. DICOM upload parsing through the full `/predict`/`/explain`
       endpoints (not just the internal `_load_dicom_slice` helper).
    3. Error status codes for invalid payloads: unsupported extensions,
       empty uploads, and corrupted/malformed file contents.

All fixtures build in-memory DICOM/NumPy payloads (`_synthetic_dicom_bytes`/
`_npy_bytes`) rather than depending on real clinical files on disk, so this
suite is fully self-contained and deterministic.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import qknee.api.server as server_module

pytestmark = [pytest.mark.slow]


# --------------------------------------------------------------------------- #
# Fixtures: live and mock backends, swapped into the module-level `backend`
# --------------------------------------------------------------------------- #

@pytest.fixture
def live_client(pca_artifact_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient wired to a real `QKneeBackend` (fitted PCA artifact,
    randomly-initialized VQC — no checkpoint needed to exercise the API
    contract itself)."""
    live_backend = server_module.QKneeBackend(pca_artifact_path=pca_artifact_path)
    assert live_backend.backend_ready, f"expected live backend to load: {live_backend.load_error}"
    monkeypatch.setattr(server_module, "backend", live_backend)
    return TestClient(server_module.app)


@pytest.fixture
def mock_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient wired to a `QKneeBackend` that deliberately fails to
    load (nonexistent PCA artifact), forcing the deterministic mock path."""
    mock_backend = server_module.QKneeBackend(pca_artifact_path=tmp_path / "no_such_artifact.pkl")
    assert not mock_backend.backend_ready
    monkeypatch.setattr(server_module, "backend", mock_backend)
    return TestClient(server_module.app)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _synthetic_dicom_bytes(pixel_array: np.ndarray) -> bytes:
    """Builds minimal, valid, in-memory DICOM bytes wrapping `pixel_array`,
    for exercising the API's DICOM upload path without a real clinical file."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    dataset = Dataset()
    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    dataset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = dataset.file_meta.MediaStorageSOPInstanceUID
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows, dataset.Columns = pixel_array.shape
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixel_array.astype(np.uint16).tobytes()

    buffer = io.BytesIO()
    pydicom.filewriter.dcmwrite(buffer, dataset, enforce_file_format=True)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# 1. /health and /predict payload shape/contents
# --------------------------------------------------------------------------- #

class TestHealthEndpoint:
    def test_health_reports_live_backend(self, live_client: TestClient):
        response = live_client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["backend_ready"] is True
        assert payload["detail"] is None

    def test_health_reports_mock_backend(self, mock_client: TestClient):
        response = mock_client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["backend_ready"] is False
        assert payload["detail"] is not None


class TestPredictEndpointPayload:
    def test_predict_with_npy_slice_live(self, live_client: TestClient, dummy_slice_2d: np.ndarray):
        response = live_client.post(
            "/predict",
            files={"file": ("slice.npy", _npy_bytes(dummy_slice_2d), "application/octet-stream")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert 0.0 <= payload["risk_score"] <= 1.0
        assert payload["diagnosis"] in {"Tear Detected", "Normal"}
        assert payload["backend"] == "live"
        assert isinstance(payload["gradcam_heatmap"], str) and len(payload["gradcam_heatmap"]) > 0

        # The heatmap must be valid, decodable base64-PNG bytes.
        import base64
        decoded = base64.b64decode(payload["gradcam_heatmap"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_predict_with_npy_slice_mock(self, mock_client: TestClient, dummy_slice_2d: np.ndarray):
        response = mock_client.post(
            "/predict",
            files={"file": ("slice.npy", _npy_bytes(dummy_slice_2d), "application/octet-stream")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["backend"] == "mock"
        assert 0.0 <= payload["risk_score"] <= 1.0

    def test_predict_with_3d_npy_volume_uses_central_slice(self, live_client: TestClient):
        rng = np.random.default_rng(10)
        volume = rng.integers(0, 255, size=(9, 64, 64), dtype=np.uint8)

        response = live_client.post(
            "/predict",
            files={"file": ("volume.npy", _npy_bytes(volume), "application/octet-stream")},
        )
        assert response.status_code == 200

    def test_predict_with_4d_multiframe_color_npy(self, live_client: TestClient):
        """Regression test: 4D (D, H, W, C) arrays are decomposed via
        DataIngestion rather than rejected outright."""
        rng = np.random.default_rng(11)
        volume = rng.integers(0, 255, size=(5, 64, 64, 3), dtype=np.uint8)

        response = live_client.post(
            "/predict",
            files={"file": ("multiframe.npy", _npy_bytes(volume), "application/octet-stream")},
        )
        assert response.status_code == 200


# --------------------------------------------------------------------------- #
# 1b. /explain payload shape/contents
# --------------------------------------------------------------------------- #

class TestExplainEndpoint:
    def test_explain_with_npy_slice_live(self, live_client: TestClient, dummy_slice_2d: np.ndarray):
        response = live_client.post(
            "/explain",
            files={"file": ("slice.npy", _npy_bytes(dummy_slice_2d), "application/octet-stream")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert set(payload.keys()) == {"gradcam_heatmap", "risk_score", "backend"}
        assert 0.0 <= payload["risk_score"] <= 1.0
        assert payload["backend"] == "live"

        import base64
        decoded = base64.b64decode(payload["gradcam_heatmap"])
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes

    def test_explain_with_npy_slice_mock(self, mock_client: TestClient, dummy_slice_2d: np.ndarray):
        response = mock_client.post(
            "/explain",
            files={"file": ("slice.npy", _npy_bytes(dummy_slice_2d), "application/octet-stream")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["backend"] == "mock"
        assert 0.0 <= payload["risk_score"] <= 1.0

    def test_explain_with_valid_dicom(self, live_client: TestClient):
        pytest.importorskip("pydicom")
        rng = np.random.default_rng(14)
        pixel_array = rng.integers(0, 4000, size=(64, 64)).astype(np.uint16)

        response = live_client.post(
            "/explain",
            files={"file": ("slice.dcm", _synthetic_dicom_bytes(pixel_array), "application/dicom")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert 0.0 <= payload["risk_score"] <= 1.0

    def test_explain_matches_predict_for_the_same_upload(self, live_client: TestClient, dummy_slice_2d: np.ndarray):
        """/explain shares /predict's exact inference call internally — the
        heatmap and risk score for the same upload must agree exactly."""
        predict_response = live_client.post(
            "/predict",
            files={"file": ("slice.npy", _npy_bytes(dummy_slice_2d), "application/octet-stream")},
        )
        explain_response = live_client.post(
            "/explain",
            files={"file": ("slice.npy", _npy_bytes(dummy_slice_2d), "application/octet-stream")},
        )

        assert predict_response.status_code == explain_response.status_code == 200
        predict_payload = predict_response.json()
        explain_payload = explain_response.json()
        assert explain_payload["risk_score"] == predict_payload["risk_score"]
        assert explain_payload["gradcam_heatmap"] == predict_payload["gradcam_heatmap"]
        assert explain_payload["backend"] == predict_payload["backend"]

    def test_explain_with_unsupported_extension_returns_415(self, live_client: TestClient):
        response = live_client.post(
            "/explain",
            files={"file": ("scan.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 415

    def test_explain_with_empty_file_returns_400(self, live_client: TestClient):
        response = live_client.post(
            "/explain",
            files={"file": ("empty.npy", b"", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_explain_with_corrupted_npy_returns_422(self, live_client: TestClient):
        response = live_client.post(
            "/explain",
            files={"file": ("broken.npy", b"this is not a numpy file", "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_explain_with_no_file_field_returns_422(self, live_client: TestClient):
        response = live_client.post("/explain")
        assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 2. DICOM parsing through the full /predict endpoint
# --------------------------------------------------------------------------- #

class TestPredictDicomParsing:
    def test_predict_with_valid_dicom(self, live_client: TestClient):
        pytest.importorskip("pydicom")
        rng = np.random.default_rng(12)
        pixel_array = rng.integers(0, 4000, size=(64, 64)).astype(np.uint16)

        response = live_client.post(
            "/predict",
            files={"file": ("slice.dcm", _synthetic_dicom_bytes(pixel_array), "application/dicom")},
        )

        assert response.status_code == 200
        payload = response.json()
        assert 0.0 <= payload["risk_score"] <= 1.0
        assert payload["backend"] == "live"

    def test_predict_with_dicom_dotdicom_extension(self, live_client: TestClient):
        """The `.dicom` extension (not just `.dcm`) must also be accepted."""
        pytest.importorskip("pydicom")
        rng = np.random.default_rng(13)
        pixel_array = rng.integers(0, 4000, size=(64, 64)).astype(np.uint16)

        response = live_client.post(
            "/predict",
            files={"file": ("slice.dicom", _synthetic_dicom_bytes(pixel_array), "application/dicom")},
        )
        assert response.status_code == 200

    def test_predict_with_malformed_dicom_bytes_returns_422(self, live_client: TestClient):
        response = live_client.post(
            "/predict",
            files={"file": ("broken.dcm", b"not a real dicom file", "application/dicom")},
        )
        assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 3. Error status codes for invalid payloads
# --------------------------------------------------------------------------- #

class TestPredictErrorHandling:
    def test_predict_with_unsupported_extension_returns_415(self, live_client: TestClient):
        response = live_client.post(
            "/predict",
            files={"file": ("scan.txt", b"hello world", "text/plain")},
        )
        assert response.status_code == 415

    def test_predict_with_empty_file_returns_400(self, live_client: TestClient):
        response = live_client.post(
            "/predict",
            files={"file": ("empty.npy", b"", "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_predict_with_corrupted_npy_returns_422(self, live_client: TestClient):
        response = live_client.post(
            "/predict",
            files={"file": ("broken.npy", b"this is not a numpy file", "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_predict_with_1d_array_returns_422(self, live_client: TestClient):
        response = live_client.post(
            "/predict",
            files={"file": ("bad_shape.npy", _npy_bytes(np.arange(10)), "application/octet-stream")},
        )
        assert response.status_code == 422

    def test_predict_with_no_file_field_returns_422(self, live_client: TestClient):
        """FastAPI's own request validation (missing the required `file` field)."""
        response = live_client.post("/predict")
        assert response.status_code == 422
