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

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving figures to disk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from quantum_dim_reduction import QuantumDimReducer
from vqc_classifier import VQCClassifier

OUTPUT_DIR = Path("eval_outputs")


@dataclass
class ModelMetrics:
    name: str
    y_true: np.ndarray
    y_prob: np.ndarray
    roc_auc: float
    sensitivity: float
    specificity: float
    f1: float


def generate_synthetic_dataset(
    n_samples: int = 800, n_features: int = 512, seed: int = 0
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


def compute_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> ModelMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return ModelMetrics(
        name=name,
        y_true=y_true,
        y_prob=y_prob,
        roc_auc=roc_auc_score(y_true, y_prob),
        sensitivity=sensitivity,
        specificity=specificity,
        f1=f1_score(y_true, y_pred),
    )


def train_svm_baseline(X_train, y_train, X_test) -> np.ndarray:
    """SVM (RBF kernel) trained directly on standardized 512-D features."""
    scaler = StandardScaler().fit(X_train)
    # SVC's own probability=True is deprecated; calibrate an SVC decision
    # function into probabilities explicitly instead.
    model = CalibratedClassifierCV(SVC(kernel="rbf", random_state=0), method="sigmoid", ensemble=False)
    model.fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def train_resnet_linear_baseline(X_train, y_train, X_test) -> np.ndarray:
    """'ResNet-only' baseline: a linear (logistic regression) probe directly
    on the 512-D ResNet features, with no PCA or quantum stage."""
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=2000, random_state=0)
    model.fit(scaler.transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_test))[:, 1]


def train_quantum_vqc(
    X_train, y_train, X_test, n_epochs: int = 60, lr: float = 0.05
) -> np.ndarray:
    """StandardScaler -> PCA(4) -> [0, 2*pi] -> 4-qubit VQC, trained with a
    standard PyTorch optimizer loop."""
    reducer = QuantumDimReducer(use_incremental_pca=False)
    quantum_train = reducer.fit_transform(X_train)
    quantum_test = reducer.transform(X_test)

    model = VQCClassifier(n_qubits=4, n_layers=3)
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


def plot_confusion_matrices(results: list[ModelMetrics], output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        y_pred = (result.y_prob >= 0.5).astype(int)
        cm = confusion_matrix(result.y_true, y_pred, labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["No Tear", "Tear"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(result.name)

    fig.suptitle("Confusion Matrices — Q-Knee Model Comparison")
    fig.tight_layout()

    output_path = output_dir / "confusion_matrices.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_roc_curves(results: list[ModelMetrics], output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))

    for result in results:
        fpr, tpr, _ = roc_curve(result.y_true, result.y_prob)
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=result.roc_auc, name=result.name).plot(ax=ax)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curves — Q-Knee Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_path = output_dir / "roc_curves.png"
    fig.savefig(output_path, dpi=150)
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


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Generating synthetic ResNet-feature dataset (swap in real embeddings + labels)...")
    features, labels = generate_synthetic_dataset(n_samples=800, n_features=512, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, stratify=labels, random_state=0
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
        compute_metrics("SVM", y_test, svm_probs),
        compute_metrics("ResNet-only", y_test, resnet_linear_probs),
        compute_metrics("Quantum VQC", y_test, vqc_probs),
    ]

    print("\n=== Evaluation Results ===")
    print_results_table(results)

    cm_path = plot_confusion_matrices(results, OUTPUT_DIR)
    roc_path = plot_roc_curves(results, OUTPUT_DIR)

    print(f"\nSaved confusion matrices to {cm_path.resolve()}")
    print(f"Saved ROC curves to {roc_path.resolve()}")
