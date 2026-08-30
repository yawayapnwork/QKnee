"""
Evaluation harness: compares the Q-Knee quantum pipeline (PCA(4) -> 4-qubit
VQC) against two classical baselines on a binary tear-classification task:

    1. SVM        - sklearn SVC(kernel="rbf") trained directly on
                    standardized 512-D ResNet features.
    2. ResNet-only - a logistic-regression linear probe trained directly on
                    the 512-D ResNet features (i.e. "just add a linear head
                    to ResNet", no PCA/quantum stage).
    3. Quantum VQC - StandardScaler -> PCA(4) -> [0, 2*pi] angle scaling
                    (quantum_dim_reduction.QuantumDimReducer) -> 4-qubit
                    PennyLane VQC (vqc_classifier.VQCClassifier), trained
                    with a standard PyTorch Adam/BCELoss loop.

Reports ROC-AUC, sensitivity (recall of the positive/"tear" class),
specificity, and F1-score for each model on a held-out test split, and
saves Confusion Matrix + ROC Curve figures to `eval_outputs/`.

No real labeled MRI dataset is bundled with this repo, so `__main__`
generates a synthetic 512-D feature dataset with a latent linear signal
(same generative scheme used to smoke-test quantum_dim_reduction.py) purely
to exercise the full evaluation pipeline end-to-end. Swap
`generate_synthetic_dataset()` for real ResNet18 embeddings + clinical
labels for an actual model comparison.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving figures to disk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qknee.config.loader import load_config
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.vqc import VQCClassifier

_config = load_config()
OUTPUT_DIR = _config.paths.eval_output_dir
RISK_THRESHOLD = _config.api.tear_risk_threshold

# Where the comparative benchmark suite's structured results/plot land —
# `qknee/artifacts/` (not `eval_output_dir`) since these are meant to be
# read by the Streamlit dashboard and the pitch deck, not just eyeballed
# during a training run.
DEFAULT_ARTIFACTS_DIR = Path("qknee/artifacts")
BENCHMARK_RESULTS_FILENAME = "benchmark_results.json"
BENCHMARK_ROC_FILENAME = "benchmark_roc_curve.png"

PredictSingleFn = Callable[[np.ndarray], float]  # (D,) feature row -> P(positive class)


@dataclass
class ModelMetrics:
    name: str
    y_true: np.ndarray
    y_prob: np.ndarray
    roc_auc: float
    sensitivity: float  # = recall of the positive ("tear") class
    specificity: float
    f1: float
    precision: float = 0.0
    recall: float = 0.0  # alias of `sensitivity`, kept as its own field for
    # callers that want the standard "Precision/Recall" naming rather than
    # the clinical "Sensitivity/Specificity" naming — both describe the
    # same held-out predictions, computed once in `_compute()`.
    confusion_matrix: List[List[int]] = field(default_factory=list)  # [[tn, fp], [fn, tp]]
    latency_ms_per_sample: Optional[float] = None  # None if not benchmarked


def generate_synthetic_dataset(
    n_samples: int = _config.evaluation.synthetic_n_samples,
    n_features: int = _config.resnet.feature_dim,
    seed: int = _config.evaluation.random_seed,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic stand-in for (ResNet18 512-D embeddings, binary tear label).

    A 4-D latent signal drives both the embedding structure (via a random
    projection, mirroring how real anatomical variation would show up
    across many correlated ResNet channels) and the label (via a linear
    decision boundary in that latent space), so the classical/quantum
    models have genuine, learnable signal to separate.
    """
    rng = np.random.default_rng(seed)

    latent = rng.normal(size=(n_samples, 4))
    projection = rng.normal(size=(4, n_features))
    noise = rng.normal(scale=1.0, size=(n_samples, n_features))
    features = latent @ projection + noise

    decision_weights = rng.normal(size=4)
    logits = latent @ decision_weights
    probabilities = 1 / (1 + np.exp(-logits))
    labels = (rng.uniform(size=n_samples) < probabilities).astype(np.int64)

    return features.astype(np.float32), labels


class PerformanceEvaluator:
    """Computes ROC-AUC, sensitivity, specificity, and F1-score for a single
    model's predictions against held-out ground-truth labels.

    Works for any binary classifier's predicted probabilities, including the
    Quantum VQC's sigmoid output (`model_pipeline.QKneeModel` / `vqc_classifier.VQCClassifier`)
    and the classical SVM/ResNet-linear baselines below.

    Usage:
        evaluator = PerformanceEvaluator("Quantum VQC", y_test, vqc_probs)
        print(evaluator.metrics)          # ModelMetrics(...)
        print(evaluator.summary())        # one-line human-readable report
    """

    def __init__(
        self,
        name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = RISK_THRESHOLD,
        latency_ms_per_sample: Optional[float] = None,
    ):
        self.name = name
        self.y_true = np.asarray(y_true)
        self.y_prob = np.asarray(y_prob)
        self.threshold = threshold
        self.latency_ms_per_sample = latency_ms_per_sample
        self.metrics = self._compute()

    def _compute(self) -> ModelMetrics:
        y_pred = (self.y_prob >= self.threshold).astype(int)
        cm = confusion_matrix(self.y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return ModelMetrics(
            name=self.name,
            y_true=self.y_true,
            y_prob=self.y_prob,
            roc_auc=roc_auc_score(self.y_true, self.y_prob),
            sensitivity=sensitivity,
            specificity=specificity,
            f1=f1_score(self.y_true, y_pred),
            precision=precision_score(self.y_true, y_pred, zero_division=0),
            recall=recall_score(self.y_true, y_pred, zero_division=0),
            confusion_matrix=cm.tolist(),
            latency_ms_per_sample=self.latency_ms_per_sample,
        )

    @property
    def roc_auc(self) -> float:
        return self.metrics.roc_auc

    @property
    def sensitivity(self) -> float:
        return self.metrics.sensitivity

    @property
    def specificity(self) -> float:
        return self.metrics.specificity

    @property
    def f1(self) -> float:
        return self.metrics.f1

    def summary(self) -> str:
        return (
            f"{self.name}: ROC-AUC={self.roc_auc:.3f} "
            f"Sensitivity={self.sensitivity:.3f} "
            f"Specificity={self.specificity:.3f} "
            f"F1={self.f1:.3f}"
        )


def compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = RISK_THRESHOLD,
    latency_ms_per_sample: Optional[float] = None,
) -> ModelMetrics:
    """Functional wrapper around `PerformanceEvaluator` (kept for callers
    that just want the metrics without instantiating the class directly)."""
    return PerformanceEvaluator(name, y_true, y_prob, threshold=threshold, latency_ms_per_sample=latency_ms_per_sample).metrics


def train_svm_baseline(X_train, y_train, X_test) -> np.ndarray:
    """SVM (RBF kernel) trained directly on standardized 512-D features."""
    scaler = StandardScaler().fit(X_train)
    # SVC's own probability=True is deprecated; calibrate an SVC decision
    # function into probabilities explicitly instead.
    model = CalibratedClassifierCV(
        SVC(kernel="rbf", random_state=_config.evaluation.random_seed), method="sigmoid", ensemble=False
    )
    model.fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def train_resnet_linear_baseline(X_train, y_train, X_test) -> np.ndarray:
    """'ResNet-only' baseline: a linear (logistic regression) probe directly
    on the 512-D ResNet features, with no PCA or quantum stage."""
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(
        max_iter=_config.evaluation.classical_max_iter, random_state=_config.evaluation.random_seed
    )
    model.fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def train_quantum_vqc(
    X_train, y_train, X_test,
    n_epochs: int = _config.training.n_epochs,
    lr: float = _config.training.learning_rate,
) -> np.ndarray:
    """StandardScaler -> PCA(n_qubits) -> [0, 2*pi] -> n-qubit VQC, trained
    with a standard PyTorch optimizer loop."""
    reducer = QuantumDimReducer(use_incremental_pca=_config.pca.use_incremental_pca)
    quantum_train = reducer.fit_transform(X_train)
    quantum_test = reducer.transform(X_test)

    model = VQCClassifier(n_qubits=_config.quantum.n_qubits, n_layers=_config.quantum.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.from_numpy(quantum_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_fn(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"    [VQC] epoch {epoch:3d} | loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(quantum_test).float()
        return model(X_test_t).squeeze(1).numpy()


# --------------------------------------------------------------------------- #
# Comparative benchmark suite: three literally-specified architectures
#   1. Pure Classical Backbone + Linear Classifier: ResNet18 -> 4-dim
#      Linear bottleneck -> Linear(4, 2) -> Softmax
#   2. Classical Backbone + SVM: ResNet18 -> PCA(4) -> RBF SVM
#   3. Hybrid Q-Knee: ResNet18 -> PCA/Bottleneck(4) -> 4-Qubit VQC
# All three funnel through the *same* 512-D ResNet18 embedding and the
# *same* 4-D bottleneck width, so the comparison isolates "what happens to
# the 4-D representation" (a linear head / an RBF kernel / a quantum
# circuit) rather than confounding it with different input dimensionality.
# --------------------------------------------------------------------------- #

class LinearBottleneckClassifier(nn.Module):
    """`ResNet18 -> Linear(feature_dim, 4) -> Linear(4, 2) -> Softmax`.

    The "Pure Classical Backbone + Linear Classifier" baseline: a trainable
    4-D linear bottleneck (the classical analogue of `QuantumDimReducer`'s
    PCA(4) or the VQC's 4 qubits) followed by a 2-class linear readout.
    Returns raw logits from `forward()`; callers apply `softmax` themselves
    (`train_linear_bottleneck_classifier` below does so at inference time).
    """

    def __init__(self, feature_dim: int, bottleneck_dim: int = 4, n_classes: int = 2):
        super().__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.bottleneck(x))


def train_linear_bottleneck_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_epochs: int = _config.training.n_epochs,
    lr: float = _config.training.learning_rate,
) -> Tuple[np.ndarray, PredictSingleFn]:
    """Trains `LinearBottleneckClassifier` (ResNet18 -> 4-dim Linear ->
    Softmax) with Adam/CrossEntropyLoss.

    Returns:
        `(y_prob_test, predict_single)` — `y_prob_test` is P(positive class)
        for every row of `X_test`; `predict_single` runs one `(D,)` row
        through scaler + model for latency benchmarking.
    """
    scaler = StandardScaler().fit(X_train)
    model = LinearBottleneckClassifier(feature_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(scaler.transform(X_train)).float()
    y_train_t = torch.from_numpy(np.asarray(y_train)).long()

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"    [Linear+Softmax] epoch {epoch:3d} | loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(scaler.transform(X_test)).float()
        y_prob_test = F.softmax(model(X_test_t), dim=1)[:, 1].numpy()

    def predict_single(x_row: np.ndarray) -> float:
        with torch.no_grad():
            scaled = scaler.transform(x_row.reshape(1, -1))
            logits = model(torch.from_numpy(scaled).float())
            return float(F.softmax(logits, dim=1)[0, 1].item())

    return y_prob_test, predict_single


def train_pca_svm_classifier(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
) -> Tuple[np.ndarray, PredictSingleFn]:
    """`ResNet18 -> StandardScaler -> PCA(n_qubits) -> RBF SVM` — the
    literal "Classical Backbone + SVM" baseline, reduced to the *same* 4-D
    bottleneck width the quantum VQC uses (unlike `train_svm_baseline`
    above, which runs the SVM directly on the full 512-D embedding).

    Returns:
        `(y_prob_test, predict_single)` — see `train_linear_bottleneck_classifier`.
    """
    n_components = _config.quantum.n_qubits
    scaler = StandardScaler().fit(X_train)
    pca = PCA(n_components=n_components, random_state=_config.evaluation.random_seed)
    X_train_reduced = pca.fit_transform(scaler.transform(X_train))
    X_test_reduced = pca.transform(scaler.transform(X_test))

    model = CalibratedClassifierCV(
        SVC(kernel="rbf", random_state=_config.evaluation.random_seed), method="sigmoid", ensemble=False
    )
    model.fit(X_train_reduced, y_train)
    y_prob_test = model.predict_proba(X_test_reduced)[:, 1]

    def predict_single(x_row: np.ndarray) -> float:
        scaled = scaler.transform(x_row.reshape(1, -1))
        reduced = pca.transform(scaled)
        return float(model.predict_proba(reduced)[0, 1])

    return y_prob_test, predict_single


def train_hybrid_qknee_vqc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_epochs: int = _config.training.n_epochs,
    lr: float = _config.training.learning_rate,
) -> Tuple[np.ndarray, PredictSingleFn]:
    """`ResNet18 -> QuantumDimReducer (StandardScaler -> PCA(4) -> [0,2*pi])
    -> 4-Qubit PennyLane VQC` — the Hybrid Q-Knee model. Same underlying
    training as `train_quantum_vqc` above; also returns a single-sample
    predict function for latency benchmarking.

    Returns:
        `(y_prob_test, predict_single)` — see `train_linear_bottleneck_classifier`.
    """
    reducer = QuantumDimReducer(use_incremental_pca=_config.pca.use_incremental_pca)
    quantum_train = reducer.fit_transform(X_train)
    quantum_test = reducer.transform(X_test)

    model = VQCClassifier(n_qubits=_config.quantum.n_qubits, n_layers=_config.quantum.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.from_numpy(quantum_train).float()
    y_train_t = torch.from_numpy(np.asarray(y_train)).float().unsqueeze(1)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_fn(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"    [Hybrid VQC] epoch {epoch:3d} | loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(quantum_test).float()
        y_prob_test = model(X_test_t).squeeze(1).numpy()

    def predict_single(x_row: np.ndarray) -> float:
        with torch.no_grad():
            angles = reducer.transform(x_row.reshape(1, -1))
            return float(model(torch.from_numpy(angles).float())[0, 0].item())

    return y_prob_test, predict_single


def measure_latency_ms_per_sample(
    predict_single: PredictSingleFn,
    X_test: np.ndarray,
    n_repeats: int = 30,
    warmup: int = 3,
    seed: int = _config.evaluation.random_seed,
) -> float:
    """Times `predict_single` on individual (batch-size-1) rows drawn from
    `X_test`, returning the mean wall-clock latency in milliseconds per
    sample. Single-sample (not batched) timing is deliberate: it reflects
    the dashboard/API's actual one-slice-at-a-time inference latency, not
    a best-case batched throughput number.

    Args:
        predict_single: `(D,) -> float` prediction function, as returned by
            `train_linear_bottleneck_classifier`/`train_pca_svm_classifier`/
            `train_hybrid_qknee_vqc`.
        X_test: Rows to sample from for timing.
        n_repeats: Number of timed calls to average over.
        warmup: Untimed calls run first, so lazy first-call overhead
            (JIT/graph compilation, PennyLane QNode tracing) doesn't
            inflate the reported latency.
        seed: RNG seed for which rows are sampled (deterministic).
    """
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, X_test.shape[0], size=warmup + n_repeats)

    for i in indices[:warmup]:
        predict_single(X_test[i])

    durations_ms = []
    for i in indices[warmup:]:
        start = time.perf_counter()
        predict_single(X_test[i])
        durations_ms.append((time.perf_counter() - start) * 1000)

    return float(np.mean(durations_ms))


def build_mrnet_validation_subset(
    root: Union[str, Path],
    resnet_extractor: Optional["ResNet18FeatureExtractor"] = None,  # noqa: F821 - see local import below
    plane: str = "sagittal",
    condition: str = "acl",
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds a real `(N, 512)` ResNet18-embedding dataset (plus binary
    labels) from an MRNet-shaped directory tree — either the real Stanford
    MRNet dataset laid out the same way, or the mock tree from
    `qknee.data.dataset.generate_mock_mrnet_dataset` (used by
    `scripts/run_benchmark.py` when no real dataset root is given).

    For each case, the anatomical-midpoint slice along `plane` (via
    `qknee.data.ingestion.MultiPlaneViewSelector`) is run through the same
    224x224/ImageNet-normalized preprocessing (`DataIngestion.preprocess`)
    and frozen `ResNet18FeatureExtractor` the live pipeline uses — so these
    features are directly comparable to (and swappable with) real
    inference-time embeddings, unlike `generate_synthetic_dataset`'s
    unrelated Gaussian features.

    Args:
        root: MRNet-shaped dataset root (contains `{split}/{plane}/*.npy`
            and `{split}-{condition}.csv`).
        resnet_extractor: A frozen `ResNet18FeatureExtractor` to reuse
            (avoids re-downloading/re-initializing pretrained weights per
            call); built fresh if omitted.
        plane: Which anatomical plane's volumes to read.
        condition: Which label CSV to read (`{split}-{condition}.csv`).
        split: Dataset split subdirectory/CSV prefix.

    Returns:
        `(features, labels)`: `features` is `(N, 512)` float32,
        `labels` is `(N,)` int64 in `{0, 1}`.

    Raises:
        FileNotFoundError: if the label CSV or a case's `.npy` volume is missing.
    """
    from qknee.data.ingestion import DataIngestion, MultiPlaneViewSelector
    from qknee.models.resnet_extractor import ResNet18FeatureExtractor

    root = Path(root)
    label_csv_path = root / f"{split}-{condition}.csv"
    if not label_csv_path.exists():
        raise FileNotFoundError(
            f"No label CSV found at {label_csv_path}. Expected an MRNet-shaped root "
            "(see qknee.data.dataset.generate_mock_mrnet_dataset for the exact layout)."
        )

    case_rows = [
        line.split(",") for line in label_csv_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    if resnet_extractor is None:
        resnet_extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    resnet_extractor.eval()

    ingestion = DataIngestion(train=False)
    features: List[np.ndarray] = []
    labels: List[int] = []

    with torch.no_grad():
        for case_id, label_str in case_rows:
            npy_path = root / split / plane / f"{case_id}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"Missing volume for case '{case_id}': {npy_path}")

            volume = np.load(npy_path)
            representative_slice = MultiPlaneViewSelector(volume).get_slice(plane)  # anatomical midpoint

            batch = ingestion.preprocess(representative_slice)  # (1, 1, 3, 224, 224)
            feature_vector = resnet_extractor(batch.squeeze(0))  # (1, 512)

            features.append(feature_vector.squeeze(0).numpy())
            labels.append(int(label_str))

    return np.stack(features).astype(np.float32), np.array(labels, dtype=np.int64)


def export_benchmark_results_json(
    results: List[ModelMetrics],
    output_path: Union[str, Path] = DEFAULT_ARTIFACTS_DIR / BENCHMARK_RESULTS_FILENAME,
    dataset_info: Optional[dict] = None,
) -> Path:
    """Serializes the comparative benchmark's per-model metrics (ROC-AUC,
    F1, precision, recall, confusion matrix, inference latency) plus a
    generation timestamp into one structured JSON file, for the Streamlit
    dashboard and hackathon pitch deck to read without re-running training.

    Args:
        results: One `ModelMetrics` per compared model.
        output_path: Destination `.json` path (parent directories created
            if missing); defaults to `qknee/artifacts/benchmark_results.json`.
        dataset_info: Optional free-form dict describing the evaluation
            dataset (e.g. `{"source": "mock", "n_train": ..., "n_test": ...}`),
            recorded alongside the per-model results for provenance.

    Returns:
        `output_path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "risk_threshold": RISK_THRESHOLD,
        "dataset": dataset_info or {},
        "models": [
            {
                "name": result.name,
                "roc_auc": result.roc_auc,
                "f1_score": result.f1,
                "precision": result.precision,
                "recall": result.recall,
                "sensitivity": result.sensitivity,
                "specificity": result.specificity,
                "confusion_matrix": {
                    "labels": ["No Tear", "Tear"],
                    "matrix": result.confusion_matrix,  # [[tn, fp], [fn, tp]]
                },
                "latency_ms_per_sample": result.latency_ms_per_sample,
                "n_test_samples": int(len(result.y_true)),
            }
            for result in results
        ],
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


def plot_confusion_matrices(results: list[ModelMetrics], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        y_pred = (result.y_prob >= RISK_THRESHOLD).astype(int)
        cm = confusion_matrix(result.y_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["No Tear", "Tear"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(result.name)

    fig.suptitle("Confusion Matrices — Q-Knee Model Comparison")
    fig.tight_layout()

    output_path = output_dir / "confusion_matrices.png"
    fig.savefig(output_path, dpi=_config.evaluation.figure_dpi)
    plt.close(fig)
    return output_path


def plot_roc_curves(results: list[ModelMetrics], output_dir: Path, filename: str = "roc_curves.png") -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))

    for result in results:
        fpr, tpr, _ = roc_curve(result.y_true, result.y_prob)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=result.roc_auc, name=result.name).plot(ax=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves — Q-Knee Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=_config.evaluation.figure_dpi)
    plt.close(fig)
    return output_path


def print_results_table(results: list[ModelMetrics]) -> None:
    header = f"{'Model':<18}{'ROC-AUC':>10}{'Sensitivity':>14}{'Specificity':>14}{'F1-score':>10}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.name:<18}{result.roc_auc:>10.3f}{result.sensitivity:>14.3f}"
            f"{result.specificity:>14.3f}{result.f1:>10.3f}"
        )


def print_benchmark_table(results: list[ModelMetrics]) -> None:
    """Like `print_results_table`, plus Precision, Recall, and Inference
    Latency columns — the full metric set `scripts/run_benchmark.py` reports.
    Model-name column width adapts to the longest name so long, fully
    descriptive model names (e.g. "Hybrid Q-Knee (ResNet18->PCA(4)->4-Qubit
    VQC)") don't break column alignment."""
    name_width = max(len("Model"), *(len(result.name) for result in results)) + 2
    header = (
        f"{'Model':<{name_width}}{'ROC-AUC':>9}{'F1':>8}{'Precision':>11}{'Recall':>9}{'Latency(ms)':>13}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        latency = f"{result.latency_ms_per_sample:.3f}" if result.latency_ms_per_sample is not None else "n/a"
        print(
            f"{result.name:<{name_width}}{result.roc_auc:>9.3f}{result.f1:>8.3f}"
            f"{result.precision:>11.3f}{result.recall:>9.3f}{latency:>13}"
        )


if __name__ == "__main__":
    torch.manual_seed(_config.evaluation.random_seed)
    np.random.seed(_config.evaluation.random_seed)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating synthetic ResNet-feature dataset (swap in real embeddings + labels)...")
    features, labels = generate_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=_config.evaluation.test_size,
        stratify=labels,
        random_state=_config.evaluation.random_seed,
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples "
          f"| positive rate: {labels.mean():.2f}\n")

    print("Training SVM baseline...")
    svm_probs = train_svm_baseline(X_train, y_train, X_test)

    print("Training ResNet-only (logistic regression probe) baseline...")
    resnet_linear_probs = train_resnet_linear_baseline(X_train, y_train, X_test)

    print("Training Quantum VQC pipeline...")
    vqc_probs = train_quantum_vqc(X_train, y_train, X_test)

    results = [
        PerformanceEvaluator("SVM", y_test, svm_probs).metrics,
        PerformanceEvaluator("ResNet-only", y_test, resnet_linear_probs).metrics,
        PerformanceEvaluator("Quantum VQC", y_test, vqc_probs).metrics,
    ]

    print("\n=== Evaluation Results ===")
    print_results_table(results)

    cm_path = plot_confusion_matrices(results, OUTPUT_DIR)
    roc_path = plot_roc_curves(results, OUTPUT_DIR)

    print(f"\nSaved confusion matrices to {cm_path.resolve()}")
    print(f"Saved ROC curves to {roc_path.resolve()}")
