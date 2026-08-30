"""
Tests for the RSNA Knee competition-standard macro-averaged ROC-AUC
additions to `qknee.models.evaluate`:

    1. `generate_multilabel_synthetic_dataset` — shape, binary labels,
       determinism, and per-condition independence.
    2. `compute_macro_auc` — the official Final Score formula, the core
       ligament/meniscal subset breakdown, and graceful handling of a
       degenerate (single-class) condition.
    3. `run_kaggle_macro_auc_benchmark` / `export_kaggle_benchmark_summary`
       end-to-end: real training across all 12 conditions for all three
       named architectures, and a structured JSON export.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qknee.data.dataset import RSNA_TARGET_COLUMNS
from qknee.models.evaluate import (
    RSNA_CORE_SUBSET,
    compute_macro_auc,
    export_kaggle_benchmark_summary,
    generate_multilabel_synthetic_dataset,
    run_kaggle_macro_auc_benchmark,
    train_quantum_vqc,
    train_resnet_linear_baseline,
    train_svm_baseline,
)

pytestmark = [pytest.mark.slow]


# --------------------------------------------------------------------------- #
# 1. generate_multilabel_synthetic_dataset
# --------------------------------------------------------------------------- #

class TestGenerateMultilabelSyntheticDataset:
    def test_shapes_and_dtypes(self):
        features, labels = generate_multilabel_synthetic_dataset(n_samples=50, n_features=64, seed=0)
        assert features.shape == (50, 64)
        assert features.dtype == np.float32
        assert set(labels.keys()) == set(RSNA_TARGET_COLUMNS)
        for array in labels.values():
            assert array.shape == (50,)
            assert set(np.unique(array)) <= {0, 1}

    def test_deterministic_given_seed(self):
        features_a, labels_a = generate_multilabel_synthetic_dataset(n_samples=40, n_features=32, seed=7)
        features_b, labels_b = generate_multilabel_synthetic_dataset(n_samples=40, n_features=32, seed=7)
        np.testing.assert_array_equal(features_a, features_b)
        for condition in RSNA_TARGET_COLUMNS:
            np.testing.assert_array_equal(labels_a[condition], labels_b[condition])

    def test_different_seeds_differ(self):
        _, labels_a = generate_multilabel_synthetic_dataset(n_samples=60, n_features=32, seed=1)
        _, labels_b = generate_multilabel_synthetic_dataset(n_samples=60, n_features=32, seed=2)
        assert any(not np.array_equal(labels_a[c], labels_b[c]) for c in RSNA_TARGET_COLUMNS)

    def test_conditions_are_not_identical_to_each_other(self):
        _, labels = generate_multilabel_synthetic_dataset(n_samples=100, n_features=32, seed=0)
        assert not np.array_equal(labels["ACL"], labels["MCL"])
        assert not np.array_equal(labels["Medial Meniscus"], labels["Fracture"])

    def test_respects_custom_condition_subset(self):
        subset = ("ACL", "MCL")
        _, labels = generate_multilabel_synthetic_dataset(n_samples=20, n_features=16, condition_names=subset, seed=0)
        assert set(labels.keys()) == set(subset)


# --------------------------------------------------------------------------- #
# 2. compute_macro_auc
# --------------------------------------------------------------------------- #

class TestComputeMacroAuc:
    @pytest.fixture
    def perfect_predictions(self):
        rng = np.random.default_rng(0)
        y_true = {c: rng.integers(0, 2, size=40) for c in RSNA_TARGET_COLUMNS}
        y_prob = {c: y_true[c].astype(float) for c in RSNA_TARGET_COLUMNS}  # perfect separation
        return y_true, y_prob

    def test_perfect_predictions_score_1(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        result = compute_macro_auc(y_true, y_prob)
        assert result["final_score"] == pytest.approx(1.0)
        assert all(v == pytest.approx(1.0) for v in result["per_condition_auc"].values())

    def test_final_score_is_mean_of_twelve(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        result = compute_macro_auc(y_true, y_prob)
        manual_mean = float(np.mean(list(result["per_condition_auc"].values())))
        assert result["final_score"] == pytest.approx(manual_mean)
        assert result["n_conditions_total"] == 12
        assert result["n_conditions_scored"] == 12

    def test_core_subset_matches_expected_conditions(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        result = compute_macro_auc(y_true, y_prob)
        assert result["core_subset"]["conditions"] == list(RSNA_CORE_SUBSET)
        expected = float(np.mean([result["per_condition_auc"][c] for c in RSNA_CORE_SUBSET]))
        assert result["core_subset"]["mean_auc"] == pytest.approx(expected)

    def test_single_class_condition_excluded_not_raised(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        y_true = dict(y_true)
        y_true["Fracture"] = np.zeros(40, dtype=int)  # single class -> AUC undefined

        result = compute_macro_auc(y_true, y_prob)  # must not raise

        assert result["per_condition_auc"]["Fracture"] is None
        assert result["n_conditions_scored"] == 11
        assert "Fracture" not in [k for k, v in result["per_condition_auc"].items() if v is not None and k == "Fracture"]

    def test_all_conditions_degenerate_gives_none_final_score(self):
        y_true = {c: np.zeros(10, dtype=int) for c in RSNA_TARGET_COLUMNS}
        y_prob = {c: np.random.default_rng(0).uniform(size=10) for c in RSNA_TARGET_COLUMNS}
        result = compute_macro_auc(y_true, y_prob)
        assert result["final_score"] is None
        assert result["core_subset"]["mean_auc"] is None


# --------------------------------------------------------------------------- #
# 3. run_kaggle_macro_auc_benchmark / export_kaggle_benchmark_summary
# --------------------------------------------------------------------------- #

class TestRunKaggleMacroAucBenchmark:
    @staticmethod
    @pytest.fixture(scope="class")
    def small_benchmark_results():
        """Runs the full 3-architecture x 12-condition benchmark once per
        test class (kept small: 80 samples, 5 VQC epochs) — real training,
        just fast."""
        return run_kaggle_macro_auc_benchmark(n_samples=80, n_epochs=5, seed=0)

    def test_covers_all_three_architectures(self, small_benchmark_results):
        assert set(small_benchmark_results.keys()) == {
            "Baseline Classical ResNet18", "Classical ResNet18 + RBF SVM", "Hybrid Q-Knee VQC",
        }

    def test_each_architecture_scores_all_twelve_conditions(self, small_benchmark_results):
        for arch_result in small_benchmark_results.values():
            assert arch_result["n_conditions_total"] == 12
            assert set(arch_result["per_condition_auc"].keys()) == set(RSNA_TARGET_COLUMNS)

    def test_final_scores_are_valid_auc_range(self, small_benchmark_results):
        for arch_result in small_benchmark_results.values():
            assert arch_result["final_score"] is not None
            assert 0.0 <= arch_result["final_score"] <= 1.0

    def test_custom_architecture_subset(self):
        custom = {"Baseline Classical ResNet18": train_resnet_linear_baseline}
        results = run_kaggle_macro_auc_benchmark(n_samples=60, architectures=custom, seed=0)
        assert set(results.keys()) == {"Baseline Classical ResNet18"}


class TestExportKaggleBenchmarkSummary:
    def test_writes_valid_json_with_expected_schema(self, tmp_path: Path):
        results = run_kaggle_macro_auc_benchmark(n_samples=60, n_epochs=5, seed=0)
        output_path = tmp_path / "kaggle_benchmark_summary.json"

        returned_path = export_kaggle_benchmark_summary(
            results, output_path=output_path, dataset_info={"source": "test"},
        )

        assert returned_path == output_path
        assert output_path.exists()

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["condition_names"] == list(RSNA_TARGET_COLUMNS)
        assert payload["core_subset"] == list(RSNA_CORE_SUBSET)
        assert payload["dataset"]["source"] == "test"
        assert set(payload["models"].keys()) == {
            "Baseline Classical ResNet18", "Classical ResNet18 + RBF SVM", "Hybrid Q-Knee VQC",
        }
        for arch_result in payload["models"].values():
            assert "final_score" in arch_result
            assert "per_condition_auc" in arch_result
            assert len(arch_result["per_condition_auc"]) == 12

    def test_creates_parent_directories(self, tmp_path: Path):
        results = run_kaggle_macro_auc_benchmark(
            n_samples=50, n_epochs=3, seed=0,
            architectures={"Baseline Classical ResNet18": train_resnet_linear_baseline},
        )
        output_path = tmp_path / "nested" / "dir" / "summary.json"
        export_kaggle_benchmark_summary(results, output_path=output_path)
        assert output_path.exists()
