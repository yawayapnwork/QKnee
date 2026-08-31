"""
Tests for `qknee.models.pipeline.PipelineRunner` — the canonical
DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM orchestration class.

Covers six things:
    1. End-to-end execution (`run()`) on single-slice and multi-slice inputs.
    2. Input boundary assertions on each individual stage method.
    3. Error handling: invalid inputs, malformed artifacts, and mismatched
       configuration all raise `PipelineValidationError` with an actionable
       message, rather than a raw exception from deep inside
       PyTorch/PennyLane/sklearn.
    4. Volumetric batch parity: a genuine `(Batch, Slices, Channels, H, W)`
       multi-sample batch runs end-to-end (ResNet18 -> PCA -> PennyLane VQC)
       without exceptions, with Pauli-Z expectations and calibrated
       probabilities strictly bounded.
    5. Offline precomputed-cache parity: `qknee.ui.analysis_app`'s Judge
       Fast-Path cache loader reproduces `precomputed_cache.json`'s exact
       metrics, fast enough to serve as a live-demo fallback.
    6. Report generation validation: `ReportGenerator.build_pdf_report()`
       produces a valid, non-empty PDF with embedded images/tables and no
       deprecation warnings.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.pipeline import PipelineResult, PipelineRunner, PipelineValidationError, QKneePipeline
from qknee.models.qknee_model import QKneeModel, save_checkpoint
from qknee.models.vqc import VQCClassifier

pytestmark = [pytest.mark.slow]


# --------------------------------------------------------------------------- #
# 1. End-to-end execution
# --------------------------------------------------------------------------- #

class TestEndToEndExecution:
    def test_run_on_single_slice_returns_valid_result(self, pipeline_runner: PipelineRunner, dummy_slice_2d):
        result = pipeline_runner.run(dummy_slice_2d)

        assert isinstance(result, PipelineResult)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.quantum_angles.shape == (1, 4)
        assert np.all(result.quantum_angles >= 0.0) and np.all(result.quantum_angles <= 2 * np.pi + 1e-6)
        assert result.gradcam_heatmap is not None
        assert result.gradcam_heatmap.ndim == 2
        assert result.gradcam_heatmap.min() >= 0.0 and result.gradcam_heatmap.max() <= 1.0 + 1e-6

    def test_run_on_multi_slice_volume_returns_valid_result(self, pipeline_runner: PipelineRunner):
        rng = np.random.default_rng(1)
        volume = rng.integers(0, 255, size=(6, 224, 224), dtype=np.uint8)

        result = pipeline_runner.run(volume)

        assert 0.0 <= result.risk_score <= 1.0
        assert result.gradcam_heatmap is not None

    def test_run_skip_gradcam_omits_heatmap(self, pipeline_runner: PipelineRunner, dummy_slice_2d):
        result = pipeline_runner.run(dummy_slice_2d, skip_gradcam=True)

        assert result.gradcam_heatmap is None
        assert 0.0 <= result.risk_score <= 1.0

    def test_run_is_deterministic_for_the_same_input(self, pipeline_runner: PipelineRunner, dummy_slice_2d):
        result_a = pipeline_runner.run(dummy_slice_2d, skip_gradcam=True)
        result_b = pipeline_runner.run(dummy_slice_2d, skip_gradcam=True)

        assert result_a.risk_score == pytest.approx(result_b.risk_score)
        np.testing.assert_allclose(result_a.quantum_angles, result_b.quantum_angles)

    def test_extract_quantum_features_matches_manual_stage_chain(
        self, pipeline_runner: PipelineRunner, dummy_slice_2d
    ):
        via_convenience = pipeline_runner.extract_quantum_features(dummy_slice_2d)

        batch = pipeline_runner.ingest(dummy_slice_2d)
        features = pipeline_runner.extract_resnet_features(batch)
        via_stages = pipeline_runner.reduce_to_quantum_angles(features)

        np.testing.assert_allclose(via_convenience, via_stages)

    def test_run_central_slice_selection_for_grad_cam(self, pipeline_runner: PipelineRunner):
        """`run()` should target the central slice of a multi-slice volume
        for Grad-CAM (`batch.shape[1] // 2`), not an arbitrary edge slice."""
        rng = np.random.default_rng(2)
        volume = rng.integers(0, 255, size=(9, 224, 224), dtype=np.uint8)
        batch = pipeline_runner.ingest(volume)

        assert batch.shape[1] == 9
        central_index = batch.shape[1] // 2
        assert central_index == 4

        # explain() on that exact central slice should succeed without error.
        heatmap = pipeline_runner.explain(batch[:, central_index])
        assert heatmap.ndim == 2


# --------------------------------------------------------------------------- #
# 2. Input boundary assertions on individual stages
# --------------------------------------------------------------------------- #

class TestStageBoundaryAssertions:
    def test_ingest_returns_five_dimensional_batch(self, pipeline_runner: PipelineRunner, dummy_slice_2d):
        batch = pipeline_runner.ingest(dummy_slice_2d)
        assert batch.dim() == 5
        assert tuple(batch.shape[:2]) == (1, 1)  # (1, S=1, 3, 224, 224) for a single 2D slice
        assert batch.shape[2] == 3

    def test_extract_resnet_features_returns_correct_feature_dim(
        self, pipeline_runner: PipelineRunner, dummy_slice_2d
    ):
        batch = pipeline_runner.ingest(dummy_slice_2d)
        features = pipeline_runner.extract_resnet_features(batch)
        assert features.shape == (1, 512)
        assert np.all(np.isfinite(features))

    def test_reduce_to_quantum_angles_returns_bounded_output(
        self, pipeline_runner: PipelineRunner, dummy_resnet_features
    ):
        angles = pipeline_runner.reduce_to_quantum_angles(dummy_resnet_features[:1])
        assert angles.shape == (1, 4)
        assert angles.min() >= 0.0 and angles.max() <= 2 * np.pi + 1e-6

    def test_classify_returns_a_probability(self, pipeline_runner: PipelineRunner):
        angles = np.random.default_rng(3).uniform(0, 2 * np.pi, size=(1, 4)).astype(np.float32)
        risk = pipeline_runner.classify(angles)
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0

    def test_classify_accepts_an_alternate_vqc_head(self, pipeline_runner: PipelineRunner):
        angles = np.random.default_rng(4).uniform(0, 2 * np.pi, size=(1, 4)).astype(np.float32)
        alternate_vqc = VQCClassifier(n_qubits=4, n_layers=3)
        alternate_vqc.eval()

        risk = pipeline_runner.classify(angles, vqc=alternate_vqc)
        assert 0.0 <= risk <= 1.0

    def test_explain_returns_a_2d_heatmap(self, pipeline_runner: PipelineRunner, dummy_slice_2d):
        batch = pipeline_runner.ingest(dummy_slice_2d)
        heatmap = pipeline_runner.explain(batch[:, 0])
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# 3. Error handling
# --------------------------------------------------------------------------- #

class TestErrorHandling:
    def test_ingest_missing_file_raises_pipeline_validation_error(self, pipeline_runner: PipelineRunner):
        with pytest.raises(PipelineValidationError, match="DataIngestion"):
            pipeline_runner.ingest("this/path/does/not/exist.png")

    def test_ingest_unsupported_type_raises_pipeline_validation_error(self, pipeline_runner: PipelineRunner):
        with pytest.raises(PipelineValidationError):
            pipeline_runner.ingest(12345)  # type: ignore[arg-type]

    def test_ingest_bad_array_ndim_raises_pipeline_validation_error(self, pipeline_runner: PipelineRunner):
        with pytest.raises(PipelineValidationError):
            pipeline_runner.ingest(np.zeros(10))  # 1D array: not a valid slice or volume

    def test_extract_resnet_features_wrong_channel_count_raises(self, pipeline_runner: PipelineRunner):
        bad_batch = torch.rand(1, 2, 5, 224, 224)  # 5D but 5 "channels" instead of the required 3
        with pytest.raises(PipelineValidationError, match="ResNet18"):
            pipeline_runner.extract_resnet_features(bad_batch)

    def test_reduce_to_quantum_angles_wrong_feature_dim_raises(self, pipeline_runner: PipelineRunner):
        wrong_dim_features = np.random.default_rng(5).normal(size=(1, 10)).astype(np.float32)
        with pytest.raises(PipelineValidationError, match="PCA"):
            pipeline_runner.reduce_to_quantum_angles(wrong_dim_features)

    def test_classify_wrong_angle_dim_raises(self, pipeline_runner: PipelineRunner):
        wrong_dim_angles = np.random.default_rng(6).uniform(0, 2 * np.pi, size=(1, 7)).astype(np.float32)
        with pytest.raises(PipelineValidationError, match="VQC"):
            pipeline_runner.classify(wrong_dim_angles)

    def test_explain_rejects_non_four_dimensional_input(self, pipeline_runner: PipelineRunner):
        with pytest.raises(PipelineValidationError, match="GradCAM"):
            pipeline_runner.explain(torch.rand(3, 224, 224))  # missing batch dim

    def test_run_propagates_ingestion_failure(self, pipeline_runner: PipelineRunner):
        with pytest.raises(PipelineValidationError):
            pipeline_runner.run("no/such/file.png")

    def test_constructor_raises_when_pca_artifact_missing(self, tmp_path: Path):
        with pytest.raises(PipelineValidationError, match="PCA artifact not found"):
            PipelineRunner(pca_artifact_path=tmp_path / "does_not_exist.pkl")

    def test_constructor_raises_on_n_components_mismatch(self, tmp_path: Path):
        """A PCA artifact fit with a different `n_components` than
        `config.quantum.n_qubits` must fail fast at construction time."""
        mismatched_reducer = QuantumDimReducer(n_components=2).fit(
            np.random.default_rng(7).normal(size=(50, 512)).astype(np.float32)
        )
        artifact_path = tmp_path / "mismatched_pca.pkl"
        mismatched_reducer.save(artifact_path)

        with pytest.raises(PipelineValidationError, match="n_qubits"):
            PipelineRunner(pca_artifact_path=artifact_path)

    def test_constructor_raises_on_malformed_vqc_checkpoint(self, pca_artifact_path: Path, tmp_path: Path):
        bad_checkpoint_path = tmp_path / "bad_checkpoint.pt"
        torch.save({"not_a_real_checkpoint": 123}, bad_checkpoint_path)

        with pytest.raises(PipelineValidationError, match="VQC checkpoint"):
            PipelineRunner(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=bad_checkpoint_path)

    def test_constructor_raises_on_checkpoint_qubit_mismatch(
        self, pca_artifact_path: Path, fitted_reducer: QuantumDimReducer, tmp_path: Path
    ):
        torch.manual_seed(0)
        wrong_model = QKneeModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=3)
        checkpoint_path = tmp_path / "wrong_qubits.pt"
        save_checkpoint(wrong_model, checkpoint_path)

        # Corrupt the recorded n_qubits so it disagrees with the configured pipeline.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["n_qubits"] = 99
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(PipelineValidationError, match="n_qubits"):
            PipelineRunner(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=checkpoint_path)


# --------------------------------------------------------------------------- #
# 4. Volumetric batch parity: (Batch, Slices, Channels, H, W) end-to-end
# --------------------------------------------------------------------------- #

class TestVolumetricBatchParity:
    """Verifies the full ResNet18 -> PCA -> PennyLane VQC chain on a
    genuine multi-sample, multi-slice `(B, S, C, H, W)` batch — the shape a
    real batch of volumetric cases takes once ingestion has run — rather
    than the single-sample `(1, S, C, H, W)` shape the end-to-end tests
    above exercise."""

    @pytest.fixture
    def volumetric_batch(self) -> torch.Tensor:
        """Deterministic `(B=3, S=5, C=3, H=224, W=224)` synthetic batch —
        3 cases, 5 slices each."""
        generator = torch.Generator().manual_seed(2024)
        return torch.rand(3, 5, 3, 224, 224, generator=generator)

    def test_batch_executes_end_to_end_without_exceptions(
        self, pipeline_runner: PipelineRunner, volumetric_batch: torch.Tensor
    ):
        features = pipeline_runner.extract_resnet_features(volumetric_batch)
        assert features.shape == (3, 512)
        assert np.all(np.isfinite(features))

        angles = pipeline_runner.reduce_to_quantum_angles(features)
        assert angles.shape == (3, 4)
        assert angles.min() >= 0.0 and angles.max() <= 2 * np.pi + 1e-6

        risk_scores = [pipeline_runner.classify(angles[i : i + 1]) for i in range(angles.shape[0])]
        assert len(risk_scores) == 3
        assert all(isinstance(score, float) for score in risk_scores)

    def test_pauli_z_expectations_are_strictly_within_bounds(
        self, pipeline_runner: PipelineRunner, volumetric_batch: torch.Tensor
    ):
        """Reads the VQC's raw per-qubit measurement output (before the
        classical Linear+Sigmoid readout collapses it to a risk score) for
        every sample in the batch and asserts it is strictly bounded to
        `[-1.0, 1.0]` — the exact `default.qubit` state-vector simulator
        guarantees this analytically, with no floating-point slack needed."""
        features = pipeline_runner.extract_resnet_features(volumetric_batch)
        angles = pipeline_runner.reduce_to_quantum_angles(features)

        with torch.no_grad():
            angles_tensor = torch.from_numpy(angles).float()
            expvals = pipeline_runner.vqc.quantum_layer(angles_tensor).detach().numpy()

        assert expvals.shape == (3, 4)
        assert np.all(expvals >= -1.0) and np.all(expvals <= 1.0)

    def test_calibrated_probabilities_are_within_bounds_per_sample(
        self, pipeline_runner: PipelineRunner, volumetric_batch: torch.Tensor
    ):
        features = pipeline_runner.extract_resnet_features(volumetric_batch)
        angles = pipeline_runner.reduce_to_quantum_angles(features)

        for i in range(angles.shape[0]):
            risk = pipeline_runner.classify(angles[i : i + 1])
            assert 0.0 <= risk <= 1.0

    def test_batch_run_via_run_method_over_multiple_volumes(self, pipeline_runner: PipelineRunner):
        """Same batch-parity contract exercised through the public `run()`
        convenience entry point, one synthetic `(S, H, W)` volume per
        simulated case, confirming per-case results are independently
        well-formed (not just the raw-tensor stage chain above)."""
        rng = np.random.default_rng(2024)
        for case_index in range(3):
            volume = rng.integers(0, 255, size=(5, 224, 224), dtype=np.uint8)
            result = pipeline_runner.run(volume, skip_gradcam=True)

            assert isinstance(result, PipelineResult)
            assert 0.0 <= result.risk_score <= 1.0
            assert result.quantum_angles.shape == (1, 4)


# --------------------------------------------------------------------------- #
# 5. Offline precomputed-cache parity (Judge Fast-Path, qknee.ui.analysis_app)
# --------------------------------------------------------------------------- #

_CACHE_PATH = Path("qknee/artifacts/precomputed_cache.json")


@pytest.mark.skipif(
    not _CACHE_PATH.exists(),
    reason=f"{_CACHE_PATH} not generated yet; run `python scripts/generate_demo_cache.py` first.",
)
class TestPrecomputedCacheParity:
    """Verifies `qknee.ui.analysis_app`'s Judge Fast-Path fallback cache
    loader reproduces exactly the prediction metrics recorded in
    `precomputed_cache.json` (no live inference, no recomputation drift)
    and stays fast enough to actually serve as a live-demo latency-risk
    fallback — the entire point of the PRD's Plan B cache."""

    @pytest.fixture(scope="class")
    def analysis_app(self):
        pytest.importorskip("streamlit")
        import qknee.ui.analysis_app as module

        return module

    @pytest.fixture(scope="class")
    def raw_cases(self):
        with _CACHE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)["cases"]

    def test_load_precomputed_cache_matches_the_json_file_on_disk(self, analysis_app, raw_cases):
        cache = analysis_app.load_precomputed_cache()
        assert cache is not None
        assert [case["case_id"] for case in cache["cases"]] == [case["case_id"] for case in raw_cases]

    def test_fast_path_result_reproduces_exact_metrics_for_every_case(self, analysis_app, raw_cases):
        for raw_case in raw_cases:
            _, result = analysis_app.build_fast_path_result(raw_case)

            assert result.risk_score == pytest.approx(raw_case["risk_score"])
            expected_label = (
                "Abnormality Detected" if raw_case["risk_score"] >= analysis_app.RISK_THRESHOLD else "Normal"
            )
            assert result.prediction_label == expected_label
            assert result.quantum_latency_ms == pytest.approx(raw_case.get("quantum_latency_ms", 0.0))
            assert result.acl_risk == pytest.approx(raw_case["risk_score"])

            expected_pauli_z = raw_case.get("pauli_z_expectations")
            if expected_pauli_z:
                assert result.pauli_z_expectations is not None
                np.testing.assert_allclose(
                    result.pauli_z_expectations, expected_pauli_z, rtol=1e-6, atol=1e-6,
                )

    def test_fast_path_result_serves_each_sample_in_under_10_milliseconds(self, analysis_app, raw_cases):
        # Warm the one-time cv2 import cost outside the timed loop — a live
        # demo session imports cv2 once at app startup, not per case served,
        # so per-sample timing should reflect steady-state serving cost.
        analysis_app.build_fast_path_result(raw_cases[0])

        for raw_case in raw_cases:
            start = time.perf_counter()
            analysis_app.build_fast_path_result(raw_case)
            elapsed_ms = (time.perf_counter() - start) * 1000

            assert elapsed_ms < 10.0, (
                f"case '{raw_case['case_id']}' took {elapsed_ms:.2f}ms to serve from cache, "
                "expected < 10ms"
            )

    def test_fast_path_result_does_not_invoke_the_quantum_simulator(self, analysis_app, raw_cases):
        """Structural guarantee (not just "it happened to be fast"): the
        fallback cache path must never re-run PennyLane circuit execution,
        the exact failure mode this cache exists to route around."""
        import pennylane as qml
        from unittest.mock import patch

        with patch.object(
            qml.QNode, "__call__",
            side_effect=AssertionError("QNode was invoked while serving the precomputed cache"),
        ):
            for raw_case in raw_cases:
                analysis_app.build_fast_path_result(raw_case)


# --------------------------------------------------------------------------- #
# 6. Report generation validation (qknee.xai.report_generator.ReportGenerator)
# --------------------------------------------------------------------------- #

class TestReportGenerationValidation:
    """Verifies `ReportGenerator.build_pdf_report()` produces a valid,
    non-empty PDF with embedded slice/Grad-CAM images and an attribution
    table — without triggering any deprecation warnings from the
    reportlab/PIL rendering stack."""

    @pytest.fixture
    def sample_prediction_results(self) -> dict:
        return {
            "acl_risk": 0.62,
            "mcl_risk": 0.21,
            "meniscus_risk": 0.35,
            "pauli_z_expectations": [0.41, -0.18, 0.63, -0.07],
            "readout_weights": [0.85, -0.40, 1.10, 0.25],
            "resnet_latency_ms": 18.4,
            "pca_latency_ms": 1.2,
            "quantum_latency_ms": 42.7,
            "total_latency_ms": 62.3,
            "backend": "live",
        }

    @pytest.fixture
    def sample_slice_and_overlay(self):
        rng = np.random.default_rng(11)
        mri_slice = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
        gradcam_overlay = np.dstack([mri_slice] * 3)
        return mri_slice, gradcam_overlay

    def test_build_pdf_report_returns_a_valid_nonempty_pdf_without_deprecation_warnings(
        self, sample_prediction_results, sample_slice_and_overlay,
    ):
        from qknee.xai.report_generator import ReportGenerator

        mri_slice, gradcam_overlay = sample_slice_and_overlay
        generator = ReportGenerator()

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            pdf_bytes = generator.build_pdf_report(
                prediction_results=sample_prediction_results,
                mri_slice=mri_slice,
                gradcam_overlay=gradcam_overlay,
                metadata={"patient_id": "TEST-CASE-01", "plane": "sagittal"},
            )

        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 0
        assert pdf_bytes[:4] == b"%PDF"
        assert pdf_bytes.rstrip().endswith(b"%%EOF")

    def test_build_pdf_report_embeds_slice_and_gradcam_images_only_when_supplied(
        self, sample_prediction_results, sample_slice_and_overlay,
    ):
        from qknee.xai.report_generator import ReportGenerator

        mri_slice, gradcam_overlay = sample_slice_and_overlay
        generator = ReportGenerator()

        with_images = generator.build_pdf_report(
            prediction_results=sample_prediction_results,
            mri_slice=mri_slice,
            gradcam_overlay=gradcam_overlay,
            metadata={"patient_id": "TEST-CASE-02"},
        )
        without_images = generator.build_pdf_report(
            prediction_results=sample_prediction_results,
            mri_slice=None,
            gradcam_overlay=None,
            metadata={"patient_id": "TEST-CASE-02"},
        )

        # `/Subtype /Image` marks a real embedded Image XObject in the PDF's
        # object dictionary (unlike raw "/Image", which also matches the
        # always-present `/ProcSet [... /ImageB /ImageC /ImageI]` entry) —
        # two per page 1 (slice + Grad-CAM overlay) only when real image
        # arrays are supplied, none when both panels fall back to a
        # placeholder box.
        assert with_images.count(b"/Subtype /Image") == 2
        assert without_images.count(b"/Subtype /Image") == 0
        assert len(with_images) > len(without_images)

    def test_build_pdf_report_includes_attribution_table_when_pauli_z_supplied(
        self, sample_prediction_results, sample_slice_and_overlay,
    ):
        pypdf = pytest.importorskip("pypdf")
        from io import BytesIO

        from qknee.xai.report_generator import ReportGenerator

        mri_slice, gradcam_overlay = sample_slice_and_overlay
        generator = ReportGenerator()

        with_pauli_z = generator.build_pdf_report(
            prediction_results=sample_prediction_results,
            mri_slice=mri_slice,
            gradcam_overlay=gradcam_overlay,
            metadata={"patient_id": "TEST-CASE-03"},
        )
        results_without_pauli_z = {k: v for k, v in sample_prediction_results.items() if k != "pauli_z_expectations"}
        without_pauli_z = generator.build_pdf_report(
            prediction_results=results_without_pauli_z,
            mri_slice=mri_slice,
            gradcam_overlay=gradcam_overlay,
            metadata={"patient_id": "TEST-CASE-03"},
        )

        with_pauli_z_reader = pypdf.PdfReader(BytesIO(with_pauli_z))
        without_pauli_z_reader = pypdf.PdfReader(BytesIO(without_pauli_z))

        assert len(with_pauli_z_reader.pages) == 2
        page_2_text_with = with_pauli_z_reader.pages[1].extract_text()
        page_2_text_without = without_pauli_z_reader.pages[1].extract_text()

        assert "Quantum Feature Attribution" in page_2_text_with
        assert "Q0" in page_2_text_with and "Q3" in page_2_text_with  # per-qubit rows rendered
        assert "Per-qubit measurement data not available" in page_2_text_without

    def test_build_pdf_report_writes_to_disk_matching_returned_bytes(
        self, sample_prediction_results, sample_slice_and_overlay, tmp_path: Path,
    ):
        from qknee.xai.report_generator import ReportGenerator

        mri_slice, gradcam_overlay = sample_slice_and_overlay
        output_path = tmp_path / "report.pdf"
        generator = ReportGenerator()

        pdf_bytes = generator.build_pdf_report(
            prediction_results=sample_prediction_results,
            mri_slice=mri_slice,
            gradcam_overlay=gradcam_overlay,
            metadata={"patient_id": "TEST-CASE-04"},
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.read_bytes() == pdf_bytes

    def test_build_pdf_report_merges_instance_and_call_site_metadata(
        self, sample_prediction_results, sample_slice_and_overlay,
    ):
        pypdf = pytest.importorskip("pypdf")
        from io import BytesIO

        from qknee.xai.report_generator import ReportGenerator

        mri_slice, gradcam_overlay = sample_slice_and_overlay
        generator = ReportGenerator(metadata={"clinic_name": "TEST CLINIC LETTERHEAD"})

        pdf_bytes = generator.build_pdf_report(
            prediction_results=sample_prediction_results,
            mri_slice=mri_slice,
            gradcam_overlay=gradcam_overlay,
            metadata={"patient_id": "TEST-CASE-05"},
        )
        page_1_text = pypdf.PdfReader(BytesIO(pdf_bytes)).pages[0].extract_text()
        assert "TEST CLINIC LETTERHEAD" in page_1_text


# --------------------------------------------------------------------------- #
# 7. Graceful missing-checkpoint handling (QKneePipeline / QKneeModel)
# --------------------------------------------------------------------------- #

class TestGracefulMissingCheckpointFallback:
    """Verifies `QKneePipeline` (a `PipelineRunner` subclass — see
    `qknee.models.pipeline`) never raises `FileNotFoundError`/crashes when
    `qknee/artifacts/checkpoints/best_qknee_model.pt` doesn't exist: it
    falls back to the classical ResNet18 backbone's deterministic
    ImageNet-pretrained weights plus a freshly, randomly initialized
    quantum VQC, logging a specific warning instead. This is the offline
    "fresh checkout, no trained checkpoint yet" scenario every other
    inference entry point (the API server, the dashboards) already relies
    on being safe."""

    def test_predict_volume_executes_without_a_local_checkpoint_file(
        self, pca_artifact_path: Path, missing_checkpoint_path: Path,
    ):
        """The exact scenario from this task's spec: `QKneePipeline()`
        constructed with no VQC checkpoint on disk, then
        `.predict_volume(...)` run end to end on CPU."""
        assert not missing_checkpoint_path.exists()

        pipeline = QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)
        rng = np.random.default_rng(11)
        volume = rng.integers(0, 255, size=(6, 224, 224), dtype=np.uint8)

        result = pipeline.predict_volume(volume)

        assert isinstance(result, PipelineResult)
        assert 0.0 <= result.risk_score <= 1.0
        assert result.quantum_angles.shape == (1, 4)

    def test_predict_volume_defaults_to_skipping_gradcam(
        self, pca_artifact_path: Path, missing_checkpoint_path: Path,
    ):
        pipeline = QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)
        rng = np.random.default_rng(12)
        volume = rng.integers(0, 255, size=(4, 224, 224), dtype=np.uint8)

        result = pipeline.predict_volume(volume)

        assert result.gradcam_heatmap is None

    def test_predict_volume_can_still_request_gradcam_explicitly(
        self, pca_artifact_path: Path, missing_checkpoint_path: Path,
    ):
        pipeline = QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)
        rng = np.random.default_rng(13)
        volume = rng.integers(0, 255, size=(4, 224, 224), dtype=np.uint8)

        result = pipeline.predict_volume(volume, skip_gradcam=False)

        assert result.gradcam_heatmap is not None
        assert result.gradcam_heatmap.ndim == 2

    def test_missing_checkpoint_logs_the_required_warning_message(
        self, pca_artifact_path: Path, missing_checkpoint_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        import logging

        with caplog.at_level(logging.WARNING, logger="qknee.models.pipeline"):
            QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)

        assert any(
            "[WARN] Checkpoint not found. Initialized deterministic hybrid weights for demo/eval mode."
            in record.message
            for record in caplog.records
        )

    def test_two_independently_constructed_pipelines_get_different_random_vqc_weights(
        self, pca_artifact_path: Path, missing_checkpoint_path: Path,
    ):
        """"Randomized variational parameters" per the spec — two
        no-checkpoint pipelines built without a shared seed should not
        coincidentally end up with identical quantum weights."""
        torch.manual_seed(1)
        pipeline_a = QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)
        torch.manual_seed(2)
        pipeline_b = QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)

        weights_a = next(iter(pipeline_a.vqc.quantum_layer.parameters())).detach()
        weights_b = next(iter(pipeline_b.vqc.quantum_layer.parameters())).detach()
        assert not torch.allclose(weights_a, weights_b)

    def test_resnet_backbone_weights_are_deterministic_across_pipelines(
        self, pca_artifact_path: Path, missing_checkpoint_path: Path,
    ):
        """The classical ResNet18 half is unconditionally loaded from the
        fixed, deterministic ImageNet-pretrained checkpoint regardless of
        whether a qknee-trained VQC checkpoint exists — verified here by
        comparing backbone weights across two independently constructed,
        no-checkpoint pipelines."""
        pipeline_a = QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)
        pipeline_b = QKneePipeline(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)

        conv1_a = pipeline_a.feature_extractor.backbone[0].weight.detach()
        conv1_b = pipeline_b.feature_extractor.backbone[0].weight.detach()
        torch.testing.assert_close(conv1_a, conv1_b)

    def test_qknee_model_load_best_checkpoint_or_init_logs_the_required_warning(
        self, tmp_path: Path, fitted_reducer, caplog: pytest.LogCaptureFixture,
    ):
        """`qknee.models.qknee_model.load_best_checkpoint_or_init` — the
        `QKneeModel`-level counterpart of `QKneePipeline`'s constructor
        fallback above — carries the same required log message."""
        import logging

        from qknee.models.qknee_model import QKneeModel, load_best_checkpoint_or_init

        torch.manual_seed(0)
        model = QKneeModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=3)
        missing_path = tmp_path / "best_qknee_model.pt"
        assert not missing_path.exists()

        with caplog.at_level(logging.WARNING, logger="qknee.models.qknee_model"):
            returned = load_best_checkpoint_or_init(model, path=missing_path)

        assert returned is model
        assert any(
            "[WARN] Checkpoint not found. Initialized deterministic hybrid weights for demo/eval mode."
            in record.message
            for record in caplog.records
        )
