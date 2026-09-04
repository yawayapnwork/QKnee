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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
from qknee.config.logging_config import get_logger
from qknee.data.dataset import RSNA_TARGET_COLUMNS
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.vqc import VQCClassifier

logger = get_logger(__name__)
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
KAGGLE_BENCHMARK_FILENAME = "kaggle_benchmark_summary.json"

# The RSNA Knee competition's core ligament/meniscal subset — a
# clinically-focused breakdown of the 12-condition macro-AUC, reported
# alongside the full Final Score by `compute_macro_auc`.
RSNA_CORE_SUBSET: Tuple[str, ...] = ("ACL", "MCL", "Medial Meniscus")

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
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = _config.training.early_stopping_min_delta,
    log_label: str = "",
) -> np.ndarray:
    """StandardScaler -> PCA(n_qubits) -> [0, 2*pi] -> n-qubit VQC, trained
    with a standard PyTorch optimizer loop.

    `n_epochs` is a ceiling, not a fixed budget, when `early_stopping_patience`
    is set: training stops early once the training-loss hasn't improved by
    more than `early_stopping_min_delta` for `early_stopping_patience`
    consecutive epochs — the same plateau contract `scripts/train.py` uses
    (`config.training.early_stopping_patience`/`early_stopping_min_delta`),
    reused here since the VQC has no separate validation split to early-stop
    against; this loop tracks the training loss itself. Every epoch's loss is
    printed (no `epoch % 20` gap), so the actual convergence shape is visible
    in the log, not just two sampled points.

    `log_label` is prefixed to each printed line (e.g. a condition name) so a
    multi-condition caller's interleaved log stays attributable.
    """
    reducer = QuantumDimReducer(use_incremental_pca=_config.pca.use_incremental_pca)
    quantum_train = reducer.fit_transform(X_train)
    quantum_test = reducer.transform(X_test)

    model = VQCClassifier(n_qubits=_config.quantum.n_qubits, n_layers=_config.quantum.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.from_numpy(quantum_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)

    prefix = f"[VQC{' ' + log_label if log_label else ''}]"
    best_loss = float("inf")
    epochs_without_improvement = 0
    stopped_epoch = n_epochs - 1

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = loss_fn(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        print(f"    {prefix} epoch {epoch:3d} | loss = {loss_value:.4f}")

        if early_stopping_patience is not None:
            if loss_value < best_loss - early_stopping_min_delta:
                best_loss = loss_value
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                stopped_epoch = epoch
                print(
                    f"    {prefix} early-stopped at epoch {epoch} "
                    f"(no improvement > {early_stopping_min_delta} for {early_stopping_patience} epochs; "
                    f"best_loss={best_loss:.4f})"
                )
                break
        else:
            stopped_epoch = epoch

    if early_stopping_patience is not None and stopped_epoch == n_epochs - 1 and epochs_without_improvement < early_stopping_patience:
        print(f"    {prefix} reached epoch ceiling ({n_epochs}) without plateauing (loss still improving)")

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


# --------------------------------------------------------------------------- #
# RSNA Knee competition-standard macro-averaged ROC-AUC
#
# The official RSNA metric averages one ROC-AUC score per target
# condition across all 12 (`qknee.data.dataset.RSNA_TARGET_COLUMNS` — the
# single source of truth for the column list, shared with
# `qknee.data.dataset.RSNAKneeDataset` and
# `scripts/generate_kaggle_submission.py`):
#
#     Final Score = (1 / 12) * sum_{i=0}^{11} AUC_i
#
# No real labeled RSNA Knee dataset ships with this repo (only a real
# *parser* for one — see `RSNAKneeDataset`), so
# `generate_multilabel_synthetic_dataset` below extends this module's
# existing `generate_synthetic_dataset` scheme (one shared 4-D latent
# embedding driving a 512-D feature space) to 12 independent per-
# condition binary labels, purely so `compute_macro_auc` and the
# comparative-analysis benchmark below are genuinely exercised end-to-end.
# Swap in real per-condition labels (e.g. from `RSNAKneeDataset.targets`)
# for an actual competition score.
# --------------------------------------------------------------------------- #

def generate_multilabel_synthetic_dataset(
    n_samples: int = _config.evaluation.synthetic_n_samples,
    n_features: int = _config.resnet.feature_dim,
    condition_names: Sequence[str] = RSNA_TARGET_COLUMNS,
    seed: int = _config.evaluation.random_seed,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Synthetic stand-in for `(ResNet18 512-D embeddings, one binary
    label per RSNA target condition)`.

    Mirrors `generate_synthetic_dataset`'s single-condition scheme: one
    shared 4-D latent embedding drives the 512-D feature structure (one
    random projection, shared across every condition — real anatomical
    variation shows up across many correlated ResNet channels for every
    condition at once) and each condition gets its *own* independent
    linear decision boundary in that latent space (deterministic per
    `(seed, condition index)`, not Python's randomized `hash()`, so this
    is reproducible across processes/runs) — conditions end up correlated
    through shared anatomy without being identical to one another.

    Args:
        n_samples: Rows to generate.
        n_features: Synthetic embedding width (matches
            `config.resnet.feature_dim`, i.e. 512).
        condition_names: Which conditions to generate a label column for;
            defaults to all 12 `RSNA_TARGET_COLUMNS`.
        seed: Base RNG seed; each condition's decision boundary is seeded
            from `seed + 1000 + index` so the whole set is reproducible
            and every condition's boundary is independent of the others'.

    Returns:
        `(features, labels)` — `features` is `(n_samples, n_features)`
        float32; `labels` is `{condition_name: (n_samples,) int64 array}`.
    """
    rng = np.random.default_rng(seed)

    latent = rng.normal(size=(n_samples, 4))
    projection = rng.normal(size=(4, n_features))
    noise = rng.normal(scale=1.0, size=(n_samples, n_features))
    features = (latent @ projection + noise).astype(np.float32)

    labels: Dict[str, np.ndarray] = {}
    for index, condition in enumerate(condition_names):
        condition_rng = np.random.default_rng(seed + 1000 + index)
        decision_weights = condition_rng.normal(size=4)
        logits = latent @ decision_weights
        probabilities = 1 / (1 + np.exp(-logits))
        labels[condition] = (condition_rng.uniform(size=n_samples) < probabilities).astype(np.int64)

    return features, labels


def build_rsna_feature_dataset(
    csv_path: Union[str, Path],
    series_dir: Union[str, Path],
    series_csv_path: Optional[Union[str, Path]] = None,
    resnet_extractor: Optional["ResNet18FeatureExtractor"] = None,  # noqa: F821 - see local import below
    condition_names: Sequence[str] = RSNA_TARGET_COLUMNS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Builds a real `(N, 512)` ResNet18-embedding dataset (plus one binary
    label array per `condition_names`) from a real, on-disk
    `qknee.data.dataset.RSNAKneeDataset` — the RSNA-competition analogue of
    `build_mrnet_validation_subset`, feeding `run_kaggle_macro_auc_benchmark`
    real data instead of `generate_multilabel_synthetic_dataset`.

    Only studies with every one of `condition_names` labeled (non-NaN) are
    kept — the real competition labels only a small minority of studies
    (see `RSNAKneeDataset`'s docstring) — so all 12 conditions share one
    consistent set of rows/train-test split, matching
    `run_kaggle_macro_auc_benchmark`'s "same split for every architecture"
    contract.

    For each kept study, every available plane's DICOM series directory is
    preprocessed via `DataIngestion.preprocess` — the whole directory of
    real slices, the same pattern `extras/scripts/generate_kaggle_submission.py`'s
    `RSNAInferenceDataset` already uses — then passed to `resnet_extractor`
    as its native `(1, S, 3, 224, 224)` shape so `ResNet18FeatureExtractor.
    forward()` dispatches to `forward_volume` (mean-pools over the S real
    slices into one `(1, 512)` embedding); squeezing the slice dimension
    away first would misdispatch to `forward_slice` and silently return S
    *unpooled* per-slice embeddings instead. Every series' embedding for a
    study is then mean-pooled into one 512-D study-level feature vector.

    This deliberately does NOT go through `MultiPlaneViewSelector` — that
    class's axis-based plane selection is built for one canonical `(D,H,W)`
    volume, and is the source of the bug flagged in
    `build_mrnet_validation_subset` (see that function's docstring); it
    doesn't apply to, and must not be introduced into, RSNA's
    directory-of-real-slices series format.

    Args:
        csv_path: Path to `train.csv` (or a subset of it).
        series_dir: Root directory containing one subdirectory per
            `StudyInstanceUID` (each holding `SeriesInstanceUID`
            subdirectories of `.dcm` files) — see `RSNAKneeDataset`.
        series_csv_path: Path to the companion `train_series.csv`;
            defaults per `RSNAKneeDataset`'s own convention.
        resnet_extractor: A frozen `ResNet18FeatureExtractor` to reuse;
            built fresh if omitted.
        condition_names: Which of the 12 `RSNA_TARGET_COLUMNS` to require
            labels for and build a label array for.

    Returns:
        `(features, labels, kept_study_uids)`: `features` is `(N, 512)`
        float32; `labels` is `{condition: (N,) int64 array}`;
        `kept_study_uids` is the `StudyInstanceUID` for each row, in order.

    Raises:
        RuntimeError: if zero studies have both full label coverage for
            `condition_names` and at least one usable DICOM series on disk.
    """
    from qknee.data.dataset import RSNAKneeDataset
    from qknee.data.ingestion import DataIngestion
    from qknee.models.resnet_extractor import ResNet18FeatureExtractor

    dataset = RSNAKneeDataset(csv_path, series_dir, series_csv_path=series_csv_path, require_targets=True)

    if resnet_extractor is None:
        resnet_extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    resnet_extractor.eval()
    ingestion = DataIngestion(train=False)

    features: List[np.ndarray] = []
    labels: Dict[str, List[int]] = {condition: [] for condition in condition_names}
    kept_uids: List[str] = []

    with torch.no_grad():
        for record in dataset:
            if record.targets is None or any(record.targets.get(condition) is None for condition in condition_names):
                continue  # unlabeled (or partially labeled) study — skip

            series_dirs = [d for dirs in record.plane_series_dirs.values() for d in dirs]
            if not series_dirs:
                logger.warning(
                    "Study %s: labeled but no usable DICOM series on disk; skipping.",
                    record.study_instance_uid,
                )
                continue

            series_features: List[np.ndarray] = []
            for series_path in series_dirs:
                try:
                    batch = ingestion.preprocess(series_path)      # (1, S, 3, 224, 224) — S real slices
                    feature_vector = resnet_extractor(batch)       # (1, 512) — forward_volume mean-pools over S
                except Exception as exc:  # noqa: BLE001 - one bad series must not drop the whole study
                    logger.warning(
                        "Study %s: failed to load/embed series at %s: %s",
                        record.study_instance_uid, series_path, exc,
                    )
                    continue
                series_features.append(feature_vector.squeeze(0).numpy())

            if not series_features:
                continue

            features.append(np.mean(series_features, axis=0).astype(np.float32))
            for condition in condition_names:
                labels[condition].append(int(record.targets[condition]))
            kept_uids.append(record.study_instance_uid)

    if not features:
        raise RuntimeError(
            "build_rsna_feature_dataset: zero labeled studies had a usable DICOM series on disk — "
            f"checked {len(dataset)} studies from {csv_path} against {series_dir}."
        )

    features_arr = np.stack(features).astype(np.float32)
    labels_arr = {condition: np.array(values, dtype=np.int64) for condition, values in labels.items()}
    return features_arr, labels_arr, kept_uids


def compute_macro_auc(
    y_true: Dict[str, np.ndarray],
    y_prob: Dict[str, np.ndarray],
    condition_names: Sequence[str] = RSNA_TARGET_COLUMNS,
    core_subset: Sequence[str] = RSNA_CORE_SUBSET,
) -> Dict[str, Any]:
    """Computes the official RSNA Knee macro-averaged ROC-AUC:

        Final Score = (1 / 12) * sum_{i=0}^{11} AUC_i

    plus a breakdown over the core ligament/meniscal subset (ACL, MCL,
    Medial Meniscus) and the full per-condition `AUC_i` table.

    Args:
        y_true: `{condition_name: (n,) binary ground-truth array}`.
        y_prob: `{condition_name: (n,) predicted-probability array}`,
            same keys/lengths as `y_true`.
        condition_names: The 12 conditions to average over (order doesn't
            affect the score; kept for a stable `per_condition_auc` key
            order in the returned dict).
        core_subset: Conditions to additionally report a sub-average for.

    Returns:
        `{"per_condition_auc": {name: float | None}, "final_score": float | None,
          "n_conditions_scored": int, "n_conditions_total": int,
          "core_subset": {"conditions": [...], "mean_auc": float | None}}`

        A condition is `None` in `per_condition_auc` (and excluded from
        both averages, with a logged warning) if its `y_true` has only one
        class present — ROC-AUC is mathematically undefined there
        (`sklearn.metrics.roc_auc_score` itself raises `ValueError`), and
        silently substituting 0.5 or dropping it from the divisor without
        saying so would misrepresent the score.
    """
    per_condition_auc: Dict[str, Optional[float]] = {}
    for condition in condition_names:
        true_labels = np.asarray(y_true[condition])
        probs = np.asarray(y_prob[condition])
        if len(np.unique(true_labels)) < 2:
            logger.warning(
                "compute_macro_auc: condition '%s' has only one class present in y_true "
                "(n=%d) — ROC-AUC is undefined; excluding it from the macro average.",
                condition, len(true_labels),
            )
            per_condition_auc[condition] = None
            continue
        per_condition_auc[condition] = float(roc_auc_score(true_labels, probs))

    valid_scores = [score for score in per_condition_auc.values() if score is not None]
    final_score = float(np.mean(valid_scores)) if valid_scores else None

    core_scores = [per_condition_auc[c] for c in core_subset if per_condition_auc.get(c) is not None]
    core_subset_mean = float(np.mean(core_scores)) if core_scores else None

    return {
        "per_condition_auc": per_condition_auc,
        "final_score": final_score,
        "n_conditions_scored": len(valid_scores),
        "n_conditions_total": len(condition_names),
        "core_subset": {
            "conditions": list(core_subset),
            "mean_auc": core_subset_mean,
        },
    }


# --------------------------------------------------------------------------- #
# Known ground-truth label-quality issues in the 58 real labeled RSNA studies.
#
# Not a code defect — a property of the source labels themselves, confirmed by
# a manual audit of all 58 studies' report text against their assigned label
# (see rsna_effusion_audit_context.txt / rsna_effusion_audit_input.csv in the
# repo root). Recorded here so `compute_macro_auc_excluding` can attach a
# reason automatically instead of a caller re-deriving/re-typing it, and so
# the reason travels with any exported summary that uses it.
# --------------------------------------------------------------------------- #

KNOWN_LABEL_ISSUES: Dict[str, str] = {
    "Effusion": (
        "Ground-truth Effusion labels were audited against report text for all 58 "
        "labeled RSNA studies. ~43% (25/58) show the same mild/small/trace-severity "
        "wording mapped to opposite labels elsewhere in the set, including 5 cases of "
        "verbatim-identical phrasing (e.g. 'Small joint effusion', 'Leve derrame "
        "articular', 'Geringer Gelenkerguss') labeled both 0 and 1 in different "
        "studies. A macro-AUC computed with Effusion included is therefore partly "
        "scoring against label noise, not just model quality."
    ),
}


def compute_macro_auc_excluding(
    per_condition_auc: Dict[str, Optional[float]],
    exclude: Sequence[str],
    core_subset: Sequence[str] = RSNA_CORE_SUBSET,
    known_issues: Dict[str, str] = KNOWN_LABEL_ISSUES,
) -> Dict[str, Any]:
    """Recomputes a macro-AUC *view* from an ALREADY-COMPUTED `per_condition_auc`
    dict (e.g. `compute_macro_auc`'s output) with `exclude` additionally dropped
    from the average — no retraining or re-scoring, just a different aggregation
    over the same per-condition scores already computed. This is a sensitivity
    check to be reported *alongside* the original full-condition view, not a
    replacement for it — see `export_kaggle_benchmark_summary`'s `alternate_views`.

    Args:
        per_condition_auc: `{condition: auc | None}`, as returned inside
            `compute_macro_auc`'s result dict.
        exclude: Conditions to drop from this view's average (in addition to
            any already `None` from `compute_macro_auc`, e.g. single-class
            test folds).
        core_subset: Same core-subset conditions `compute_macro_auc` reports
            on; any of these in `exclude` is dropped from this view's core
            subset too.
        known_issues: `{condition: reason}` — a reason string is attached
            per excluded condition when present here (see `KNOWN_LABEL_ISSUES`);
            an excluded condition absent from this dict gets no reason text,
            not an error, since not every exclusion is label-quality-driven.

    Returns:
        `{"excluded_conditions": [...], "exclusion_reasons": {condition: reason},
          "final_score": float | None, "n_conditions_scored": int,
          "core_subset": {"conditions": [...], "mean_auc": float | None}}`
    """
    exclude_set = set(exclude)
    kept = {
        condition: score
        for condition, score in per_condition_auc.items()
        if score is not None and condition not in exclude_set
    }
    final_score = float(np.mean(list(kept.values()))) if kept else None

    kept_core = [c for c in core_subset if c not in exclude_set]
    core_scores = [per_condition_auc[c] for c in kept_core if per_condition_auc.get(c) is not None]
    core_mean = float(np.mean(core_scores)) if core_scores else None

    return {
        "excluded_conditions": list(exclude),
        "exclusion_reasons": {c: known_issues[c] for c in exclude if c in known_issues},
        "final_score": final_score,
        "n_conditions_scored": len(kept),
        "core_subset": {
            "conditions": kept_core,
            "mean_auc": core_mean,
        },
    }


# Which callable implements each named architecture for the comparative
# analysis below — reuses this module's three *original* baseline
# functions (each trained directly on the full 512-D embedding, no 4-D
# PCA/quantum bottleneck for the two classical models), matching this
# section's literal architecture names.
_KAGGLE_BENCHMARK_ARCHITECTURES: Dict[str, Callable] = {
    "Baseline Classical ResNet18": train_resnet_linear_baseline,
    "Classical ResNet18 + RBF SVM": train_svm_baseline,
    "Hybrid Q-Knee VQC": train_quantum_vqc,
}


def run_kaggle_macro_auc_benchmark(
    n_samples: int = _config.evaluation.synthetic_n_samples,
    n_epochs: int = 20,
    seed: int = _config.evaluation.random_seed,
    condition_names: Sequence[str] = RSNA_TARGET_COLUMNS,
    architectures: Optional[Dict[str, Callable]] = None,
    rsna_csv_path: Optional[Union[str, Path]] = None,
    rsna_series_dir: Optional[Union[str, Path]] = None,
    rsna_series_csv_path: Optional[Union[str, Path]] = None,
    vqc_early_stopping_patience: Optional[int] = None,
    vqc_early_stopping_min_delta: float = _config.training.early_stopping_min_delta,
) -> Dict[str, Any]:
    """Trains each named architecture once per RSNA target condition and
    computes each architecture's macro-averaged ROC-AUC (`compute_macro_auc`)
    — the comparative analysis `export_kaggle_benchmark_summary` serializes.

    Data source: synthetic by default (`generate_multilabel_synthetic_dataset`,
    `n_samples` rows). Pass `rsna_csv_path` + `rsna_series_dir` (real
    `train.csv` + the `train_series/` DICOM root — see
    `qknee.data.dataset.RSNAKneeDataset`) to instead pull real, ResNet18-
    embedded studies via `build_rsna_feature_dataset` — `n_samples` is then
    ignored; the row count is however many real studies have both full
    label coverage for `condition_names` and a usable DICOM series on disk
    (the real competition labels only a small minority of studies, so this
    is typically far smaller than the synthetic default).

    All three architectures share the exact same train/test row split and
    the exact same per-condition labels, so the comparison isolates "which
    architecture" rather than confounding it with different data.

    Args:
        n_samples: Synthetic dataset size (ignored when `rsna_csv_path` is given).
        n_epochs: Training epochs for the VQC architecture (the two
            classical baselines below don't use gradient-descent epochs).
        seed: RNG seed for the synthetic dataset (when used) and the train/test split.
        condition_names: Which of the 12 RSNA conditions to score.
        architectures: `{name: train_fn}` overriding the default trio
            (`_KAGGLE_BENCHMARK_ARCHITECTURES`) — `train_fn` must have the
            signature `(X_train, y_train, X_test) -> (n_test,) probability array`.
        rsna_csv_path: Real `train.csv` path — switches to real data when given.
        rsna_series_dir: Real `train_series/` DICOM root — required with `rsna_csv_path`.
        rsna_series_csv_path: Real `train_series.csv` path; defaults per
            `RSNAKneeDataset`'s own convention if omitted.
        vqc_early_stopping_patience: If set, `n_epochs` becomes a ceiling and
            the VQC's per-condition training loop stops once its training
            loss hasn't improved by `vqc_early_stopping_min_delta` for this
            many consecutive epochs — see `train_quantum_vqc`. `None`
            (default) preserves the old fixed-`n_epochs` behavior.
        vqc_early_stopping_min_delta: Minimum training-loss decrease counted
            as an improvement, when `vqc_early_stopping_patience` is set.

    Returns:
        `{architecture_name: compute_macro_auc(...)-shaped dict}`.
    """
    architectures = architectures or _KAGGLE_BENCHMARK_ARCHITECTURES

    if rsna_csv_path is not None:
        if rsna_series_dir is None:
            raise ValueError("rsna_series_dir is required when rsna_csv_path is given.")
        features, labels, kept_uids = build_rsna_feature_dataset(
            rsna_csv_path, rsna_series_dir, series_csv_path=rsna_series_csv_path,
            condition_names=condition_names,
        )
        n_samples = len(kept_uids)
        logger.info("run_kaggle_macro_auc_benchmark: using %d real labeled RSNA studies from %s", n_samples, rsna_csv_path)
    else:
        features, labels = generate_multilabel_synthetic_dataset(
            n_samples=n_samples, condition_names=condition_names, seed=seed,
        )

    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=_config.evaluation.test_size, random_state=seed,
    )
    X_train, X_test = features[train_idx], features[test_idx]

    results: Dict[str, Any] = {}
    for arch_name, train_fn in architectures.items():
        logger.info("Training '%s' across %d RSNA conditions...", arch_name, len(condition_names))
        y_true_test: Dict[str, np.ndarray] = {}
        y_prob_test: Dict[str, np.ndarray] = {}

        for condition in condition_names:
            y_train = labels[condition][train_idx]
            y_test = labels[condition][test_idx]
            if train_fn is train_quantum_vqc:
                probs = train_fn(
                    X_train, y_train, X_test, n_epochs=n_epochs,
                    early_stopping_patience=vqc_early_stopping_patience,
                    early_stopping_min_delta=vqc_early_stopping_min_delta,
                    log_label=condition,
                )
            else:
                probs = train_fn(X_train, y_train, X_test)
            y_true_test[condition] = y_test
            y_prob_test[condition] = probs

        macro_auc_result = compute_macro_auc(y_true_test, y_prob_test, condition_names=condition_names)
        results[arch_name] = macro_auc_result
        logger.info("  '%s' Final Score = %.4f", arch_name, macro_auc_result["final_score"])

    return results


def export_kaggle_benchmark_summary(
    comparative_results: Dict[str, Any],
    output_path: Union[str, Path] = DEFAULT_ARTIFACTS_DIR / KAGGLE_BENCHMARK_FILENAME,
    dataset_info: Optional[Dict[str, Any]] = None,
    alternate_views: Optional[Dict[str, Sequence[str]]] = None,
) -> Path:
    """Serializes the RSNA Knee macro-AUC comparative-analysis results
    (from `run_kaggle_macro_auc_benchmark`) to a structured JSON file, for
    live presentation on the Streamlit dashboard.

    Args:
        comparative_results: `{architecture_name: compute_macro_auc(...)-shaped
            dict}`, as returned by `run_kaggle_macro_auc_benchmark`.
        output_path: Destination `.json` path (parent directories created
            if missing); defaults to `qknee/artifacts/kaggle_benchmark_summary.json`.
        dataset_info: Optional free-form dict describing the evaluation
            dataset (e.g. `{"source": "synthetic", "n_train": ..., "n_test": ...}`),
            recorded alongside the results for provenance.
        alternate_views: Optional `{view_name: conditions_to_exclude}` — for
            each entry, every model additionally gets an `alternate_scores.
            {view_name}` block (via `compute_macro_auc_excluding`, no
            retraining/re-scoring involved) alongside its original full-
            condition `per_condition_auc`/`final_score`/`core_subset`, which
            are left completely untouched. Use this for a sensitivity check
            (e.g. dropping a condition whose ground-truth labels were audited
            and found inconsistent — see `KNOWN_LABEL_ISSUES`) that should be
            reported *next to*, not instead of, the original numbers.

    Returns:
        `output_path`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models_payload: Dict[str, Any] = {}
    for arch_name, result in comparative_results.items():
        model_entry = dict(result)  # shallow copy — never mutate the caller's dict
        if alternate_views:
            model_entry["alternate_scores"] = {
                view_name: compute_macro_auc_excluding(result["per_condition_auc"], exclude=excluded)
                for view_name, excluded in alternate_views.items()
            }
        models_payload[arch_name] = model_entry

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "RSNA Knee macro-averaged ROC-AUC (Final Score = mean of 12 per-condition AUCs)",
        "condition_names": list(RSNA_TARGET_COLUMNS),
        "core_subset": list(RSNA_CORE_SUBSET),
        "dataset": dataset_info or {},
        "models": models_payload,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Saved Kaggle benchmark summary to %s", output_path)
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
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Q-Knee evaluation: single-condition SVM/ResNet/VQC comparison + "
                     "12-condition RSNA Knee macro-AUC benchmark."
    )
    arg_parser.add_argument(
        "--rsna-csv", type=str, default=None,
        help="Real RSNA train.csv path — switches the 12-condition macro-AUC benchmark "
             "from synthetic data (generate_multilabel_synthetic_dataset) to real labeled "
             "studies (build_rsna_feature_dataset). Requires --rsna-series-dir.",
    )
    arg_parser.add_argument(
        "--rsna-series-dir", type=str,
        default=str(_config.paths.rsna_series_dir) if _config.paths.rsna_series_dir else None,
        help="Real train_series/ DICOM root (StudyInstanceUID/SeriesInstanceUID/*.dcm). "
             "Defaults to config.yaml's paths.rsna_series_dir if set.",
    )
    arg_parser.add_argument(
        "--rsna-series-csv", type=str, default=None,
        help="Real train_series.csv path; defaults to alongside --rsna-csv if omitted.",
    )
    arg_parser.add_argument(
        "--kaggle-n-epochs", type=int, default=20,
        help="Epoch ceiling for the Hybrid Q-Knee VQC in the 12-condition RSNA "
             "macro-AUC benchmark (run_kaggle_macro_auc_benchmark). The two classical "
             "baselines are epoch-independent, so this only affects the VQC. Acts as a "
             "hard cap (not a fixed budget) when --kaggle-vqc-early-stopping-patience is set.",
    )
    arg_parser.add_argument(
        "--kaggle-vqc-early-stopping-patience", type=int, default=None,
        help="If set, the VQC's per-condition training loop stops once its training loss "
             "hasn't improved by --kaggle-vqc-early-stopping-min-delta for this many "
             "consecutive epochs, instead of always running --kaggle-n-epochs. Omit to keep "
             "the old fixed-epoch behavior.",
    )
    arg_parser.add_argument(
        "--kaggle-vqc-early-stopping-min-delta", type=float,
        default=_config.training.early_stopping_min_delta,
        help="Minimum training-loss decrease counted as an improvement for "
             "--kaggle-vqc-early-stopping-patience. Defaults to config.yaml's "
             "training.early_stopping_min_delta.",
    )
    cli_args = arg_parser.parse_args()

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

    print("\n=== RSNA Knee macro-averaged ROC-AUC (12-condition Kaggle benchmark) ===")
    if cli_args.rsna_csv:
        print(f"Using REAL RSNA data: {cli_args.rsna_csv} / {cli_args.rsna_series_dir}")
    kaggle_results = run_kaggle_macro_auc_benchmark(
        seed=_config.evaluation.random_seed,
        n_epochs=cli_args.kaggle_n_epochs,
        rsna_csv_path=cli_args.rsna_csv,
        rsna_series_dir=cli_args.rsna_series_dir,
        rsna_series_csv_path=cli_args.rsna_series_csv,
        vqc_early_stopping_patience=cli_args.kaggle_vqc_early_stopping_patience,
        vqc_early_stopping_min_delta=cli_args.kaggle_vqc_early_stopping_min_delta,
    )
    for arch_name, arch_result in kaggle_results.items():
        print(
            f"{arch_name}: Final Score = {arch_result['final_score']:.4f} "
            f"(core ACL/MCL/Medial-Meniscus subset = {arch_result['core_subset']['mean_auc']:.4f})"
        )
    if cli_args.rsna_csv:
        kaggle_dataset_info = {
            "source": "real",
            "rsna_csv_path": cli_args.rsna_csv,
            "rsna_series_dir": cli_args.rsna_series_dir,
            "note": "Real labeled RSNA Knee studies via qknee.data.dataset.RSNAKneeDataset "
                    "(build_rsna_feature_dataset) — n_samples is however many studies had full "
                    "12-condition label coverage and a usable DICOM series on disk.",
        }
    else:
        kaggle_dataset_info = {
            "source": "synthetic",
            "n_samples": _config.evaluation.synthetic_n_samples,
            "note": "No --rsna-csv/--rsna-series-dir given; see "
                    "qknee.data.dataset.RSNAKneeDataset for the real-data parser.",
        }
    kaggle_summary_path = export_kaggle_benchmark_summary(
        kaggle_results,
        dataset_info=kaggle_dataset_info,
        # Effusion's ground-truth labels were manually audited (see
        # KNOWN_LABEL_ISSUES above) and found internally inconsistent on the
        # exact same report wording in ~43% of the 58 labeled studies, so its
        # macro-AUC contribution is partly noise, not model quality. Always
        # export the 9-condition view alongside the full 12-condition one
        # rather than silently dropping Effusion from the primary numbers.
        alternate_views={"excl_effusion": ["Effusion"]},
    )
    print(f"Saved Kaggle benchmark summary to {kaggle_summary_path.resolve()}")
