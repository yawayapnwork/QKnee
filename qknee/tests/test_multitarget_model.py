"""
Tests for the multi-label RSNA (12-condition) additions to
`qknee.models.qknee_model`, `qknee.models.vqc_multitarget`, and
`qknee.models.pipeline`.

Covers:
    1. `vqc_multitarget`: both head architectures (`MultiObservableVQC`,
       `EnsembleMultiTargetHead`) — output shape, raw-logit range (i.e.
       NOT pre-squashed into [0,1]), gradient flow through every
       parameter, and the config-driven `build_multi_target_head` factory.
    2. `QKneeMultiTargetModel`: end-to-end image -> 12 logits, calibrated
       `predict_proba()` in [0,1], and that both head types plug in
       identically.
    3. `compute_pos_weight` / `train_qknee_multitarget_model`: correct
       imbalance-weight formula and real BCEWithLogitsLoss training that
       decreases loss.
    4. Multi-target checkpoint save/load round-trip, including the
       head-type/target-name/architecture mismatch guards.
    5. `PipelineRunner.classify_multitarget` end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qknee.data.dataset import RSNA_TARGET_COLUMNS
from qknee.models.qknee_model import (
    QKneeMultiTargetModel,
    compute_pos_weight,
    load_multitarget_checkpoint,
    save_multitarget_checkpoint,
    train_qknee_multitarget_model,
)
from qknee.models.vqc_multitarget import (
    N_RSNA_TARGETS,
    PAULI_WORD_WIRES,
    SECONDARY_CONDITIONS,
    TRIAD_CONDITIONS,
    EnsembleMultiTargetHead,
    MultiObservableVQC,
    build_multi_target_head,
)

pytestmark = [pytest.mark.slow]

HEAD_TYPES = ("multi_observable", "ensemble")


# --------------------------------------------------------------------------- #
# 1. vqc_multitarget heads
# --------------------------------------------------------------------------- #

class TestMultiTargetHeads:
    def test_pauli_word_wires_has_twelve_entries_over_four_qubits(self):
        assert len(PAULI_WORD_WIRES) == 12
        for wires in PAULI_WORD_WIRES:
            assert all(0 <= w < 4 for w in wires)
            assert len(set(wires)) == len(wires)  # no repeated qubit within one word

    def test_triad_and_secondary_partition_all_twelve_conditions(self):
        assert TRIAD_CONDITIONS == ("ACL", "MCL", "Medial Meniscus")
        assert len(TRIAD_CONDITIONS) + len(SECONDARY_CONDITIONS) == 12
        assert set(TRIAD_CONDITIONS) | set(SECONDARY_CONDITIONS) == set(RSNA_TARGET_COLUMNS)

    @pytest.mark.parametrize("head_type,head_cls", [("multi_observable", MultiObservableVQC), ("ensemble", EnsembleMultiTargetHead)])
    def test_factory_builds_correct_class(self, head_type, head_cls):
        head = build_multi_target_head(head_type, n_qubits=4, n_layers=2)
        assert isinstance(head, head_cls)

    def test_unknown_head_type_raises(self):
        with pytest.raises(ValueError, match="Unknown multi_target_head"):
            build_multi_target_head("not_a_real_head_type")

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_output_shape_is_batch_by_twelve(self, head_type):
        torch.manual_seed(0)
        head = build_multi_target_head(head_type, n_qubits=4, n_layers=2)
        x = torch.rand(6, 4) * 2 * torch.pi
        logits = head(x)
        assert logits.shape == (6, N_RSNA_TARGETS)

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_output_is_raw_logits_not_pre_squashed(self, head_type):
        """The head's own forward() must NOT apply sigmoid — values should
        routinely fall outside [0, 1] (a real logit range), confirming
        BCEWithLogitsLoss gets genuine raw logits, not probabilities."""
        torch.manual_seed(1)
        head = build_multi_target_head(head_type, n_qubits=4, n_layers=2)
        x = torch.rand(20, 4) * 2 * torch.pi
        logits = head(x)
        assert torch.isfinite(logits).all()
        # Statistically near-certain for freshly initialized weights across
        # 20 samples x 12 outputs; a head that (bug) returned sigmoid'd
        # probabilities could never produce a value outside [0, 1].
        assert (logits < 0.0).any() or (logits > 1.0).any()

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_sigmoid_of_output_is_valid_probability(self, head_type):
        torch.manual_seed(2)
        head = build_multi_target_head(head_type, n_qubits=4, n_layers=2)
        x = torch.rand(6, 4) * 2 * torch.pi
        probabilities = torch.sigmoid(head(x))
        assert torch.all(probabilities >= 0.0) and torch.all(probabilities <= 1.0)

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_gradient_flows_to_every_parameter(self, head_type):
        torch.manual_seed(3)
        head = build_multi_target_head(head_type, n_qubits=4, n_layers=2)
        x = torch.rand(4, 4) * 2 * torch.pi
        head(x).sum().backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"no gradient reached '{name}'"
            assert torch.isfinite(param.grad).all()

    def test_multi_observable_requires_four_qubits(self):
        with pytest.raises(ValueError):
            MultiObservableVQC(n_qubits=5)

    def test_ensemble_triad_circuits_are_independent(self):
        """Each triad sub-circuit must have its own weights — perturbing
        one shouldn't change another's output."""
        torch.manual_seed(0)
        head = EnsembleMultiTargetHead(n_qubits=4, n_layers=2)
        x = torch.rand(3, 4) * 2 * torch.pi
        before = head(x).detach().clone()

        with torch.no_grad():
            head.triad_circuits[0].readout.weight.add_(1.0)

        after = head(x).detach()
        assert not torch.allclose(before[:, 0], after[:, 0])  # ACL column changed
        assert torch.allclose(before[:, 1:], after[:, 1:])    # everything else unchanged


# --------------------------------------------------------------------------- #
# 2. QKneeMultiTargetModel
# --------------------------------------------------------------------------- #

class TestQKneeMultiTargetModel:
    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_forward_returns_raw_logits_shape(self, head_type, fitted_reducer, dummy_image_batch):
        torch.manual_seed(0)
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2, head_type=head_type)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_image_batch)
        assert logits.shape == (dummy_image_batch.shape[0], N_RSNA_TARGETS)

    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_predict_proba_is_calibrated(self, head_type, fitted_reducer, dummy_image_batch):
        torch.manual_seed(0)
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2, head_type=head_type)
        model.eval()
        with torch.no_grad():
            logits = model(dummy_image_batch)
            probabilities = model.predict_proba(dummy_image_batch)
        assert probabilities.shape == logits.shape
        assert torch.all(probabilities >= 0.0) and torch.all(probabilities <= 1.0)
        torch.testing.assert_close(probabilities, torch.sigmoid(logits))

    def test_mismatched_pca_reducer_n_components_raises(self, fitted_reducer):
        with pytest.raises(ValueError, match="n_qubits"):
            QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=5)

    def test_defaults_to_config_head_type(self, fitted_reducer):
        from qknee.config.loader import load_config

        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2)
        assert model.head_type == load_config().quantum.multi_target_head

    def test_trainable_parameters_excludes_frozen_pca_and_resnet(self, fitted_reducer):
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2)
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        assert not any(name.startswith("pca_layer.") for name in trainable_names)
        assert not any(name.startswith("resnet.") for name in trainable_names)
        assert any(name.startswith("head.") for name in trainable_names)


# --------------------------------------------------------------------------- #
# 3. compute_pos_weight / train_qknee_multitarget_model
# --------------------------------------------------------------------------- #

class TestComputePosWeight:
    def test_balanced_column_gives_weight_one(self):
        labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])  # 2 pos, 2 neg
        weight = compute_pos_weight(labels)
        assert weight.shape == (1,)
        assert weight[0] == pytest.approx(1.0)

    def test_rare_positive_class_gets_large_weight(self):
        labels = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])  # 1 pos, 9 neg
        weight = compute_pos_weight(labels)
        assert weight[0] == pytest.approx(9.0)

    def test_zero_positives_does_not_produce_inf_or_nan(self):
        labels = torch.zeros(10, 3)
        weight = compute_pos_weight(labels)
        assert torch.isfinite(weight).all()

    def test_shape_matches_number_of_columns(self):
        labels = torch.randint(0, 2, (50, N_RSNA_TARGETS)).float()
        weight = compute_pos_weight(labels)
        assert weight.shape == (N_RSNA_TARGETS,)


class TestTrainQKneeMultiTargetModel:
    def test_loss_decreases_and_uses_bce_with_logits(self, fitted_reducer, dummy_image_batch):
        torch.manual_seed(0)
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2, head_type="multi_observable")
        labels = torch.randint(0, 2, (dummy_image_batch.shape[0], N_RSNA_TARGETS)).float()

        history = train_qknee_multitarget_model(model, dummy_image_batch, labels, n_epochs=15, lr=0.1, log_every=5)

        assert len(history) == 15
        assert history[-1] < history[0]

    def test_wrong_label_width_raises(self, fitted_reducer, dummy_image_batch):
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2)
        wrong_labels = torch.randint(0, 2, (dummy_image_batch.shape[0], 5)).float()
        with pytest.raises(ValueError, match="TARGET_NAMES"):
            train_qknee_multitarget_model(model, dummy_image_batch, wrong_labels, n_epochs=1)

    def test_explicit_pos_weight_is_used_over_computed(self, fitted_reducer, dummy_image_batch):
        """Smoke test: passing an explicit pos_weight must not raise and
        must still train (a wrong-shaped explicit pos_weight would surface
        as a BCEWithLogitsLoss broadcast error)."""
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2)
        labels = torch.randint(0, 2, (dummy_image_batch.shape[0], N_RSNA_TARGETS)).float()
        explicit_pos_weight = torch.ones(N_RSNA_TARGETS) * 2.0

        history = train_qknee_multitarget_model(
            model, dummy_image_batch, labels, n_epochs=3, pos_weight=explicit_pos_weight,
        )
        assert len(history) == 3


# --------------------------------------------------------------------------- #
# 4. Multi-target checkpoint save/load
# --------------------------------------------------------------------------- #

class TestMultitargetCheckpoint:
    def test_round_trip_produces_identical_predictions(self, fitted_reducer, dummy_image_batch, tmp_path: Path):
        torch.manual_seed(0)
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2, head_type="multi_observable")
        model.eval()
        with torch.no_grad():
            pre_save = model.predict_proba(dummy_image_batch).clone()

        checkpoint_path = tmp_path / "multitarget.pt"
        save_multitarget_checkpoint(model, checkpoint_path, epoch=3)

        reloaded = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2, head_type="multi_observable")
        load_multitarget_checkpoint(reloaded, checkpoint_path)
        reloaded.eval()

        with torch.no_grad():
            post_load = reloaded.predict_proba(dummy_image_batch)

        torch.testing.assert_close(pre_save, post_load, rtol=1e-5, atol=1e-6)

    def test_missing_file_raises_file_not_found(self, fitted_reducer, tmp_path: Path):
        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2)
        with pytest.raises(FileNotFoundError):
            load_multitarget_checkpoint(model, tmp_path / "does_not_exist.pt")

    def test_single_target_checkpoint_rejected_with_clear_error(self, fitted_reducer, tmp_path: Path):
        """A qknee_model.pt-style single-target checkpoint is missing this
        schema's required keys entirely — must raise KeyError, not a
        confusing downstream failure."""
        import torch as torch_module

        bad_checkpoint_path = tmp_path / "single_target_style.pt"
        torch_module.save({"vqc_state_dict": {}, "model_state_dict": {}, "n_qubits": 4, "n_layers": 2}, bad_checkpoint_path)

        model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2)
        with pytest.raises(KeyError):
            load_multitarget_checkpoint(model, bad_checkpoint_path)

    def test_head_type_mismatch_raises(self, fitted_reducer, tmp_path: Path):
        torch.manual_seed(0)
        source_model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2, head_type="multi_observable")
        checkpoint_path = tmp_path / "checkpoint.pt"
        save_multitarget_checkpoint(source_model, checkpoint_path)

        mismatched_model = QKneeMultiTargetModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=2, head_type="ensemble")
        with pytest.raises(ValueError, match="head_type"):
            load_multitarget_checkpoint(mismatched_model, checkpoint_path)


# --------------------------------------------------------------------------- #
# 5. PipelineRunner.classify_multitarget
# --------------------------------------------------------------------------- #

class TestPipelineClassifyMultitarget:
    @pytest.mark.parametrize("head_type", HEAD_TYPES)
    def test_end_to_end_via_pipeline_runner(self, head_type, pipeline_runner, dummy_slice_2d):
        torch.manual_seed(0)
        head = build_multi_target_head(head_type, n_qubits=pipeline_runner.vqc.n_qubits, n_layers=2)

        batch = pipeline_runner.ingest(dummy_slice_2d)
        features = pipeline_runner.extract_resnet_features(batch)
        angles = pipeline_runner.reduce_to_quantum_angles(features)
        probabilities = pipeline_runner.classify_multitarget(angles, head=head)

        assert probabilities.shape == (1, N_RSNA_TARGETS)
        assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0

    def test_failing_head_raises_pipeline_validation_error(self, pipeline_runner, dummy_slice_2d):
        from qknee.models.pipeline import PipelineValidationError

        class BrokenHead(torch.nn.Module):
            def forward(self, x):
                raise RuntimeError("simulated failure")

        batch = pipeline_runner.ingest(dummy_slice_2d)
        features = pipeline_runner.extract_resnet_features(batch)
        angles = pipeline_runner.reduce_to_quantum_angles(features)

        with pytest.raises(PipelineValidationError, match="MultiTargetHead"):
            pipeline_runner.classify_multitarget(angles, head=BrokenHead())

    def test_custom_target_names_controls_expected_width(self, pipeline_runner, dummy_slice_2d):
        from qknee.models.pipeline import PipelineValidationError

        batch = pipeline_runner.ingest(dummy_slice_2d)
        features = pipeline_runner.extract_resnet_features(batch)
        angles = pipeline_runner.reduce_to_quantum_angles(features)

        head = build_multi_target_head("multi_observable", n_qubits=pipeline_runner.vqc.n_qubits, n_layers=2)
        # Head produces 12 columns; declaring only 5 expected names must
        # surface as a clear validation error, not a silent shape mismatch.
        with pytest.raises(PipelineValidationError):
            pipeline_runner.classify_multitarget(angles, head=head, target_names=["a", "b", "c", "d", "e"])
