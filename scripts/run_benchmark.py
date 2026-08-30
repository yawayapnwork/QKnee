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
from qknee.models.resnet_extractor import ResNet18FeatureExtractor

logger = get_logger(__name__)
_config = load_config()


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
) -> Path:
    """Runs the full 3-model comparative benchmark end-to-end and writes
    `benchmark_results.json` + `benchmark_roc_curve.png` to `output_dir`.

    Returns the path to the written JSON results file.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    )


if __name__ == "__main__":
    main()
