"""
Comparative benchmark runner: evaluates three models on the same MRNet-style
validation subset and exports the results for the Streamlit dashboard and
the hackathon pitch deck.

Models compared (all sharing the same 512-D ResNet18 embedding and the same
4-D bottleneck width, so the comparison isolates "what happens to the 4-D
representation"):

    1. Pure Classical Backbone + Linear Classifier
           ResNet18 -> Linear(512, 4) bottleneck -> Linear(4, 2) -> Softmax
    2. Classical Backbone + SVM
           ResNet18 -> StandardScaler -> PCA(4) -> RBF SVM
    3. Hybrid Q-Knee
           ResNet18 -> QuantumDimReducer (PCA(4) -> [0, 2*pi]) -> 4-Qubit VQC

Metrics: ROC-AUC, F1-score, Precision, Recall, Confusion Matrix, and
single-sample inference latency (ms/sample) for each model.

Dataset: by default, builds a deterministic *mock* MRNet-shaped dataset
(`qknee.data.dataset.generate_mock_mrnet_dataset`) so this runs end-to-end
without the real (multi-GB, credentialed) Stanford MRNet download. Point
`--data-root` at a real MRNet-shaped directory (`{split}/{plane}/*.npy` +
`{split}-{condition}.csv`) to benchmark on real data with the exact same
code path.

Outputs:
    qknee/artifacts/benchmark_results.json   - structured per-model metrics
    qknee/artifacts/benchmark_roc_curve.png  - ROC curve, all 3 models

Verification (complete end-to-end coverage, not just metric computation):
    Before training the comparison suite, `run_pipeline_sanity_check()`
    drives a synthetic `(Batch, Slices, Channels, H, W)` volumetric batch
    through the real `PipelineRunner` (DataIngestion -> ResNet18 -> PCA ->
    PennyLane VQC), asserting the quantum stage's Pauli-Z expectations and
    calibrated risk probabilities are strictly within their mathematical
    bounds — the same contract `qknee/tests/test_pipeline_runner.py`'s
    `TestVolumetricBatchParity` checks in CI, run here as a live smoke test
    against whatever code is currently checked out. After training,
    `verify_benchmark_results()` asserts every exported metric (ROC-AUC,
    F1, precision, recall, latency) is finite and within its valid range
    before anything is written to disk, so a broken metric fails loudly
    here rather than silently reaching the deck/dashboard. Skip either
    check with `--skip-sanity-check` / `--skip-metrics-verification`.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --data-root /path/to/real/mrnet --plane axial --n-cases 120
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow `python scripts/run_benchmark.py` to resolve the `qknee` package
# without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from qknee.config.logging_config import get_logger, setup_logging
from qknee.config.loader import load_config
from qknee.data.dataset import generate_mock_mrnet_dataset
from qknee.models.evaluate import (
    DEFAULT_ARTIFACTS_DIR,
    BENCHMARK_RESULTS_FILENAME,
    BENCHMARK_ROC_FILENAME,
    ModelMetrics,
    PerformanceEvaluator,
    build_mrnet_validation_subset,
    export_benchmark_results_json,
    measure_latency_ms_per_sample,
    plot_confusion_matrices,
    plot_roc_curves,
    print_benchmark_table,
    train_hybrid_qknee_vqc,
    train_linear_bottleneck_classifier,
    train_pca_svm_classifier,
)
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.resnet_extractor import ResNet18FeatureExtractor

logger = get_logger(__name__)
_config = load_config()


# --------------------------------------------------------------------------- #
# End-to-end verification: pipeline sanity check (pre-training smoke test)
# --------------------------------------------------------------------------- #

def run_pipeline_sanity_check(seed: int = _config.evaluation.random_seed) -> None:
    """Drives a synthetic `(Batch, Slices, Channels, H, W)` volumetric batch
    through a real `PipelineRunner` end-to-end (DataIngestion's later
    stages -> ResNet18 -> PCA -> PennyLane VQC) and asserts the quantum
    stage's outputs are strictly bounded, so a broken pipeline is caught
    before spending time training the comparison suite.

    Uses a freshly fitted `QuantumDimReducer` (persisted to a throwaway
    temp file) and a randomly initialized VQC — this is a structural smoke
    test of the pipeline's stage-to-stage contracts, not a check of any
    particular trained checkpoint's accuracy.

    Raises `RuntimeError` (with the underlying assertion message) if any
    bound is violated.
    """
    from qknee.models.pipeline import PipelineRunner

    print("\nRunning pipeline sanity check (synthetic (B, S, C, H, W) volumetric batch)...")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    reducer = QuantumDimReducer().fit(rng.normal(size=(300, _config.resnet.feature_dim)).astype(np.float32))
    with tempfile.TemporaryDirectory(prefix="qknee_sanity_check_") as tmp_dir:
        pca_artifact_path = Path(tmp_dir) / "pca_scaler.pkl"
        reducer.save(pca_artifact_path)
        missing_checkpoint_path = Path(tmp_dir) / "no_such_checkpoint.pt"

        try:
            runner = PipelineRunner(pca_artifact_path=pca_artifact_path, vqc_checkpoint_path=missing_checkpoint_path)

            batch_size, n_slices = 3, 5
            generator = torch.Generator().manual_seed(seed)
            volumetric_batch = torch.rand(batch_size, n_slices, 3, 224, 224, generator=generator)

            features = runner.extract_resnet_features(volumetric_batch)
            if features.shape != (batch_size, _config.resnet.feature_dim):
                raise RuntimeError(f"expected ResNet18 features shaped ({batch_size}, {_config.resnet.feature_dim}), got {features.shape}")

            angles = runner.reduce_to_quantum_angles(features)
            if angles.shape != (batch_size, _config.quantum.n_qubits):
                raise RuntimeError(f"expected {batch_size} x {_config.quantum.n_qubits} quantum angles, got {angles.shape}")

            with torch.no_grad():
                angles_tensor = torch.from_numpy(angles).float()
                pauli_z_expvals = runner.vqc.quantum_layer(angles_tensor).detach().numpy()
            if not (np.all(pauli_z_expvals >= -1.0) and np.all(pauli_z_expvals <= 1.0)):
                raise RuntimeError(
                    f"Pauli-Z expectation values out of [-1.0, 1.0] bounds: "
                    f"min={pauli_z_expvals.min():.6f}, max={pauli_z_expvals.max():.6f}"
                )

            risk_scores = [runner.classify(angles[i : i + 1]) for i in range(batch_size)]
            if not all(0.0 <= score <= 1.0 for score in risk_scores):
                raise RuntimeError(f"calibrated risk probabilities out of [0.0, 1.0] bounds: {risk_scores}")
        except Exception as exc:
            raise RuntimeError(f"Pipeline sanity check FAILED: {exc}") from exc

    print(
        f"  PASS — {batch_size} samples x {n_slices} slices ran end-to-end; "
        f"Pauli-Z in [{pauli_z_expvals.min():.4f}, {pauli_z_expvals.max():.4f}], "
        f"risk scores in [{min(risk_scores):.4f}, {max(risk_scores):.4f}]."
    )


def verify_benchmark_results(results: list[ModelMetrics]) -> None:
    """Asserts every exported metric is finite and within its valid range
    before `run_benchmark()` writes anything to disk — catches a silently
    broken metric (e.g. an unstratified split collapsing ROC-AUC to NaN,
    or a negative latency from a clock issue) before it reaches the
    dashboard/deck rather than after.

    Raises `RuntimeError` (naming the offending model/metric) on any
    violation.
    """
    print("\nVerifying benchmark results are finite and within valid ranges...")
    for metrics in results:
        for bounded_field in ("roc_auc", "sensitivity", "specificity", "f1", "precision", "recall"):
            value = getattr(metrics, bounded_field)
            if value is None or not np.isfinite(value) or not (0.0 <= value <= 1.0):
                raise RuntimeError(
                    f"Benchmark verification FAILED: {metrics.name}.{bounded_field} = {value!r} "
                    "is not a finite value in [0.0, 1.0]"
                )
        if metrics.latency_ms_per_sample is not None and (
            not np.isfinite(metrics.latency_ms_per_sample) or metrics.latency_ms_per_sample < 0.0
        ):
            raise RuntimeError(
                f"Benchmark verification FAILED: {metrics.name}.latency_ms_per_sample = "
                f"{metrics.latency_ms_per_sample!r} is not a finite non-negative value"
            )
        if len(metrics.y_prob) and not np.all((metrics.y_prob >= 0.0) & (metrics.y_prob <= 1.0)):
            raise RuntimeError(
                f"Benchmark verification FAILED: {metrics.name}.y_prob contains values outside [0.0, 1.0]"
            )
    print(f"  PASS — {len(results)} model(s) verified.")


def build_validation_subset(
    data_root: str | None,
    plane: str,
    condition: str,
    split: str,
    n_cases: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolves `--data-root` (a real MRNet-shaped directory) if given, else
    builds a deterministic mock one via `generate_mock_mrnet_dataset`, then
    extracts real `(N, 512)` ResNet18 features + labels from it via
    `build_mrnet_validation_subset`.

    Returns `(features, labels, dataset_info)` — `dataset_info` is recorded
    into the exported JSON for provenance (mock vs. real, case count, etc).
    """
    torch.manual_seed(seed)
    resnet_extractor = ResNet18FeatureExtractor(freeze_backbone=True)

    if data_root is not None:
        root = Path(data_root)
        logger.info("Using real MRNet-shaped dataset root: %s", root)
        source = "real"
    else:
        mock_dir = Path(tempfile.mkdtemp(prefix="qknee_mock_mrnet_"))
        case_ids = [f"{i:04d}" for i in range(n_cases)]
        root = generate_mock_mrnet_dataset(
            mock_dir, case_ids=case_ids, planes=(plane,), condition=condition,
            split=split, num_slices=8, size=64, seed=seed,
        )
        logger.info("No --data-root given; generated a mock MRNet dataset at %s (%d cases).", root, n_cases)
        source = "mock"

    features, labels = build_mrnet_validation_subset(
        root, resnet_extractor=resnet_extractor, plane=plane, condition=condition, split=split,
    )
    dataset_info = {
        "source": source,
        "plane": plane,
        "condition": condition,
        "split": split,
        "n_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "positive_rate": float(labels.mean()) if len(labels) else None,
    }
    return features, labels, dataset_info


def run_benchmark(
    data_root: str | None = None,
    plane: str = "sagittal",
    condition: str = "acl",
    split: str = "train",
    n_cases: int = 60,
    test_size: float = 0.25,
    n_epochs: int = 40,
    latency_repeats: int = 20,
    output_dir: Path = DEFAULT_ARTIFACTS_DIR,
    seed: int = _config.evaluation.random_seed,
    skip_sanity_check: bool = False,
    skip_metrics_verification: bool = False,
) -> Path:
    """Runs the full 3-model comparative benchmark end-to-end and writes
    `benchmark_results.json` + `benchmark_roc_curve.png` to `output_dir`.

    Unless disabled, brackets the benchmark with two verification passes
    (see the module docstring): `run_pipeline_sanity_check()` before
    training, `verify_benchmark_results()` after — both raise `RuntimeError`
    on failure rather than letting a broken pipeline/metric reach disk.

    Returns the path to the written JSON results file.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not skip_sanity_check:
        run_pipeline_sanity_check(seed=seed)

    logger.info("Building MRNet validation subset (plane=%s, condition=%s)...", plane, condition)
    features, labels, dataset_info = build_validation_subset(data_root, plane, condition, split, n_cases, seed)

    if len(np.unique(labels)) < 2:
        raise RuntimeError(
            f"Validation subset has only one class present ({np.unique(labels)}) — "
            "increase --n-cases so both labels appear, or check --condition."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=seed,
    )
    dataset_info["n_train"] = int(X_train.shape[0])
    dataset_info["n_test"] = int(X_test.shape[0])
    logger.info(
        "Train: %d samples | Test: %d samples | positive rate: %.2f",
        X_train.shape[0], X_test.shape[0], float(labels.mean()),
    )

    print("\nTraining Model 1/3: Pure Classical Backbone + Linear Classifier "
          "(ResNet18 -> Linear(4) -> Softmax)...")
    linear_probs, linear_predict = train_linear_bottleneck_classifier(X_train, y_train, X_test, n_epochs=n_epochs)

    print("\nTraining Model 2/3: Classical Backbone + SVM (ResNet18 -> PCA(4) -> RBF SVM)...")
    svm_probs, svm_predict = train_pca_svm_classifier(X_train, y_train, X_test)

    print("\nTraining Model 3/3: Hybrid Q-Knee (ResNet18 -> PCA(4) -> 4-Qubit VQC)...")
    vqc_probs, vqc_predict = train_hybrid_qknee_vqc(X_train, y_train, X_test, n_epochs=n_epochs)

    print("\nBenchmarking single-sample inference latency...")
    linear_latency = measure_latency_ms_per_sample(linear_predict, X_test, n_repeats=latency_repeats)
    svm_latency = measure_latency_ms_per_sample(svm_predict, X_test, n_repeats=latency_repeats)
    vqc_latency = measure_latency_ms_per_sample(vqc_predict, X_test, n_repeats=latency_repeats)

    results = [
        PerformanceEvaluator(
            "Classical Linear (ResNet18->4D Linear->Softmax)", y_test, linear_probs,
            latency_ms_per_sample=linear_latency,
        ).metrics,
        PerformanceEvaluator(
            "Classical SVM (ResNet18->PCA(4)->RBF SVM)", y_test, svm_probs,
            latency_ms_per_sample=svm_latency,
        ).metrics,
        PerformanceEvaluator(
            "Hybrid Q-Knee (ResNet18->PCA(4)->4-Qubit VQC)", y_test, vqc_probs,
            latency_ms_per_sample=vqc_latency,
        ).metrics,
    ]

    print("\n=== Comparative Benchmark Results ===")
    print_benchmark_table(results)

    if not skip_metrics_verification:
        verify_benchmark_results(results)

    output_dir = Path(output_dir)
    json_path = export_benchmark_results_json(
        results, output_path=output_dir / BENCHMARK_RESULTS_FILENAME, dataset_info=dataset_info,
    )
    roc_path = plot_roc_curves(results, output_dir, filename=BENCHMARK_ROC_FILENAME)
    cm_path = plot_confusion_matrices(results, output_dir)

    print(f"\nSaved structured results to {json_path.resolve()}")
    print(f"Saved ROC curve to {roc_path.resolve()}")
    print(f"Saved confusion matrices to {cm_path.resolve()}")

    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to a real MRNet-shaped dataset root ({split}/{plane}/*.npy + "
             "{split}-{condition}.csv). Omit to auto-generate a mock dataset.",
    )
    parser.add_argument("--plane", choices=["axial", "coronal", "sagittal"], default="sagittal")
    parser.add_argument("--condition", type=str, default="acl", help="Label CSV suffix, e.g. 'acl'/'meniscus'/'abnormal'.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n-cases", type=int, default=60, help="Cases to generate for the mock dataset (ignored with --data-root).")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-epochs", type=int, default=40, help="Training epochs for the two trainable models.")
    parser.add_argument("--latency-repeats", type=int, default=20, help="Timed single-sample calls averaged per model.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--seed", type=int, default=_config.evaluation.random_seed)
    parser.add_argument(
        "--skip-sanity-check", action="store_true",
        help="Skip the pre-training PipelineRunner (B,S,C,H,W) volumetric smoke test.",
    )
    parser.add_argument(
        "--skip-metrics-verification", action="store_true",
        help="Skip the post-training bounds-check on exported ROC-AUC/F1/precision/recall/latency.",
    )
    args = parser.parse_args()

    setup_logging()
    run_benchmark(
        data_root=args.data_root,
        plane=args.plane,
        condition=args.condition,
        split=args.split,
        n_cases=args.n_cases,
        test_size=args.test_size,
        n_epochs=args.n_epochs,
        latency_repeats=args.latency_repeats,
        output_dir=args.output_dir,
        seed=args.seed,
        skip_sanity_check=args.skip_sanity_check,
        skip_metrics_verification=args.skip_metrics_verification,
    )


if __name__ == "__main__":
    main()
