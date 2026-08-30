"""
Tests for the enhanced `scripts/train.py` CLI and the `qknee.config.loader`
dynamic-merge machinery it's built on.

Covers:
    1. `deep_merge`/`load_config_with_overrides` (dictionary-merge
       correctness, purity, and that cross-field config validation still
       runs against the merged result).
    2. `build_vqc` — all three `--ansatz` choices produce a module with
       the shared `(B, n_qubits) -> (B, 1)` interface.
    3. `resolve_dataset_dir` — `--plane` joining + graceful fallback.
    4. `build_synthetic_image_dataset` — shape/dtype/label-range for
       `--use_mock`/`--dry_run`.
    5. `run_training_loop` — real-time metrics (loss/accuracy/val ROC-AUC/
       gradient norm) are populated, checkpoints land in `checkpoint_dir`,
       and early stopping actually triggers on a stalled validation loss.
    6. `--dry_run` end-to-end via subprocess: 1 batch, 1 epoch, no
       `--dataset_dir` required, exits 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from qknee.config.loader import deep_merge, load_config, load_config_with_overrides, ConfigError
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.qknee_model import QKneeModel
from scripts.train import (
    ANSATZ_CHOICES,
    EpochMetrics,
    build_synthetic_image_dataset,
    build_vqc,
    compute_grad_norm,
    resolve_dataset_dir,
    run_training_loop,
)


# --------------------------------------------------------------------------- #
# 1. Dynamic dictionary merging
# --------------------------------------------------------------------------- #

class TestDeepMerge:
    def test_merge_overwrites_only_specified_leaf(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        merged = deep_merge(base, {"a": {"b": 99}})
        assert merged == {"a": {"b": 99, "c": 2}, "d": 3}

    def test_merge_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        overrides = {"a": {"b": 2}}
        deep_merge(base, overrides)
        assert base == {"a": {"b": 1}}
        assert overrides == {"a": {"b": 2}}

    def test_merge_adds_new_keys(self):
        merged = deep_merge({"a": 1}, {"b": 2})
        assert merged == {"a": 1, "b": 2}

    def test_merge_none_override_replaces_value(self):
        merged = deep_merge({"a": {"b": 1}}, {"a": {"b": None}})
        assert merged["a"]["b"] is None


class TestLoadConfigWithOverrides:
    def test_no_overrides_matches_load_config(self):
        base = load_config()
        overridden = load_config_with_overrides(None)
        assert overridden.training.n_epochs == base.training.n_epochs
        assert overridden.data.batch_size == base.data.batch_size

    def test_overrides_apply_to_specified_leaves_only(self):
        base = load_config()
        overridden = load_config_with_overrides({"training": {"n_epochs": 999}})
        assert overridden.training.n_epochs == 999
        assert overridden.training.learning_rate == base.training.learning_rate  # untouched

    def test_multiple_sections_merge_together(self):
        overridden = load_config_with_overrides({
            "training": {"n_epochs": 5, "learning_rate": 0.5},
            "data": {"batch_size": 8},
        })
        assert overridden.training.n_epochs == 5
        assert overridden.training.learning_rate == 0.5
        assert overridden.data.batch_size == 8

    def test_cross_field_validation_still_enforced_on_merged_config(self):
        """An override that breaks `pca.n_components == quantum.n_qubits`
        must fail loudly, not silently produce an inconsistent config."""
        with pytest.raises(ConfigError):
            load_config_with_overrides({"quantum": {"n_qubits": 8}})

    def test_result_is_not_cached_across_different_overrides(self):
        a = load_config_with_overrides({"training": {"n_epochs": 11}})
        b = load_config_with_overrides({"training": {"n_epochs": 22}})
        assert a.training.n_epochs == 11
        assert b.training.n_epochs == 22


# --------------------------------------------------------------------------- #
# 2. Ansatz factory
# --------------------------------------------------------------------------- #

class TestBuildVQC:
    @pytest.mark.parametrize("ansatz", ANSATZ_CHOICES)
    def test_each_ansatz_produces_matching_io_shape(self, ansatz):
        n_qubits, n_layers = 4, 2
        vqc = build_vqc(ansatz, n_qubits=n_qubits, n_layers=n_layers)
        x = torch.rand(3, n_qubits) * 2 * torch.pi
        output = vqc(x)
        assert output.shape == (3, 1)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

    def test_unknown_ansatz_raises(self):
        from scripts.train import TrainingError

        with pytest.raises(TrainingError):
            build_vqc("not_a_real_ansatz", n_qubits=4, n_layers=2)

    def test_each_ansatz_is_a_valid_qkneemodel_vqc(self):
        """The whole point of the shared interface: any ansatz should
        drop into QKneeModel's `vqc=` argument without special-casing."""
        rng = np.random.default_rng(0)
        reducer = QuantumDimReducer().fit(rng.normal(size=(50, 512)).astype(np.float32))
        for ansatz in ANSATZ_CHOICES:
            vqc = build_vqc(ansatz, n_qubits=4, n_layers=2)
            model = QKneeModel(pca_reducer=reducer, n_qubits=4, n_layers=2, vqc=vqc)
            model.eval()
            with torch.no_grad():
                output = model(torch.rand(2, 3, 224, 224))
            assert output.shape == (2, 1)


# --------------------------------------------------------------------------- #
# 3. --plane resolution
# --------------------------------------------------------------------------- #

class TestResolveDatasetDir:
    def test_none_plane_returns_dataset_dir_unchanged(self, tmp_path: Path):
        assert resolve_dataset_dir(tmp_path, None) == tmp_path

    def test_existing_plane_subdirectory_is_joined(self, tmp_path: Path):
        (tmp_path / "sagittal").mkdir()
        result = resolve_dataset_dir(tmp_path, "sagittal")
        assert result == tmp_path / "sagittal"

    def test_missing_plane_subdirectory_falls_back_to_dataset_dir(self, tmp_path: Path):
        result = resolve_dataset_dir(tmp_path, "coronal")
        assert result == tmp_path


# --------------------------------------------------------------------------- #
# 4. Synthetic dataset for --use_mock / --dry_run
# --------------------------------------------------------------------------- #

class TestBuildSyntheticImageDataset:
    def test_shape_and_dtype(self):
        images, labels = build_synthetic_image_dataset(n_samples=10, image_size=(32, 32), seed=0)
        assert images.shape == (10, 3, 32, 32)
        assert images.dtype == torch.float32
        assert labels.shape == (10,)
        assert labels.dtype == torch.int64

    def test_labels_are_binary(self):
        _, labels = build_synthetic_image_dataset(n_samples=50, image_size=(16, 16), seed=1)
        assert set(labels.tolist()) <= {0, 1}

    def test_deterministic_given_seed(self):
        images_a, labels_a = build_synthetic_image_dataset(n_samples=8, image_size=(16, 16), seed=5)
        images_b, labels_b = build_synthetic_image_dataset(n_samples=8, image_size=(16, 16), seed=5)
        assert torch.equal(images_a, images_b)
        assert torch.equal(labels_a, labels_b)

    def test_pixel_values_in_unit_range(self):
        images, _ = build_synthetic_image_dataset(n_samples=5, image_size=(16, 16), seed=2)
        assert images.min() >= 0.0 and images.max() <= 1.0


# --------------------------------------------------------------------------- #
# 5. Training loop: real-time metrics, checkpointing, early stopping
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tiny_qknee_model() -> QKneeModel:
    rng = np.random.default_rng(0)
    reducer = QuantumDimReducer().fit(rng.normal(size=(50, 512)).astype(np.float32))
    torch.manual_seed(0)
    return QKneeModel(pca_reducer=reducer, n_qubits=4, n_layers=2)


class TestRunTrainingLoop:
    def test_history_populates_all_required_metrics(self, tiny_qknee_model: QKneeModel, tmp_path: Path):
        images, labels = build_synthetic_image_dataset(n_samples=8, image_size=(224, 224), seed=0)
        train_images, train_labels = images[:6], labels[:6]
        eval_images, eval_labels = images[6:], labels[6:]

        history, best_path = run_training_loop(
            tiny_qknee_model, train_images, train_labels, eval_images, eval_labels,
            n_epochs=2, lr=0.05, device=torch.device("cpu"), threshold=0.5,
            checkpoint_dir=tmp_path, log_every=1,
            early_stopping_patience=10, early_stopping_min_delta=1e-4,
        )

        assert len(history) == 2
        for metrics in history:
            assert isinstance(metrics, EpochMetrics)
            assert metrics.grad_norm >= 0.0
            assert 0.0 <= metrics.train_accuracy <= 1.0
            assert metrics.val_loss is not None  # eval split was provided

    def test_checkpoints_are_written_to_checkpoint_dir(self, tiny_qknee_model: QKneeModel, tmp_path: Path):
        images, labels = build_synthetic_image_dataset(n_samples=8, image_size=(224, 224), seed=1)
        run_training_loop(
            tiny_qknee_model, images[:6], labels[:6], images[6:], labels[6:],
            n_epochs=1, lr=0.05, device=torch.device("cpu"), threshold=0.5,
            checkpoint_dir=tmp_path, log_every=1,
            early_stopping_patience=10, early_stopping_min_delta=1e-4,
        )
        checkpoint_files = list(tmp_path.glob("*.pt"))
        assert any("checkpoint_epoch" in p.name for p in checkpoint_files)
        assert (tmp_path / "best_checkpoint.pt").exists()

    def test_early_stopping_triggers_on_stalled_validation_loss(self, tiny_qknee_model: QKneeModel, tmp_path: Path):
        images, labels = build_synthetic_image_dataset(n_samples=8, image_size=(224, 224), seed=2)
        history, _ = run_training_loop(
            tiny_qknee_model, images[:6], labels[:6], images[6:], labels[6:],
            n_epochs=50, lr=0.05, device=torch.device("cpu"), threshold=0.5,
            checkpoint_dir=tmp_path, log_every=1,
            # min_delta=1.0 guarantees no epoch counts as "improved" (BCE
            # loss can't realistically drop by 1.0 in one full-batch step).
            early_stopping_patience=2, early_stopping_min_delta=1.0,
        )
        assert len(history) < 50  # stopped early, didn't run all 50 epochs

    def test_no_eval_split_skips_val_metrics_without_crashing(self, tiny_qknee_model: QKneeModel, tmp_path: Path):
        images, labels = build_synthetic_image_dataset(n_samples=4, image_size=(224, 224), seed=3)
        history, best_path = run_training_loop(
            tiny_qknee_model, images, labels, None, None,
            n_epochs=1, lr=0.05, device=torch.device("cpu"), threshold=0.5,
            checkpoint_dir=tmp_path, log_every=1,
            early_stopping_patience=10, early_stopping_min_delta=1e-4,
        )
        assert history[0].val_loss is None
        assert history[0].val_roc_auc is None
        assert best_path is None  # no eval split -> no "best" checkpoint selected


class TestComputeGradNorm:
    def test_zero_before_backward(self):
        model = torch.nn.Linear(4, 1)
        assert compute_grad_norm(model) == 0.0

    def test_positive_after_backward(self):
        model = torch.nn.Linear(4, 1)
        output = model(torch.rand(2, 4))
        output.sum().backward()
        assert compute_grad_norm(model) > 0.0


# --------------------------------------------------------------------------- #
# 6. --dry_run end-to-end (subprocess, exercises the real CLI)
# --------------------------------------------------------------------------- #

class TestDryRunSubprocess:
    def test_dry_run_exits_zero_and_reports_success(self):
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--dry_run"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True, text=True, timeout=300,
        )
        combined_output = result.stdout + result.stderr
        assert result.returncode == 0, combined_output[-3000:]
        assert "Dry run OK" in combined_output
