"""
Tests for `qknee.models.pipeline.PipelineRunner` — the canonical
DataIngestion -> ResNet18 -> PCA -> VQC -> GradCAM orchestration class.

Covers three things:
    1. End-to-end execution (`run()`) on single-slice and multi-slice inputs.
    2. Input boundary assertions on each individual stage method.
    3. Error handling: invalid inputs, malformed artifacts, and mismatched
       configuration all raise `PipelineValidationError` with an actionable
       message, rather than a raw exception from deep inside
       PyTorch/PennyLane/sklearn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.pipeline import PipelineResult, PipelineRunner, PipelineValidationError
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
