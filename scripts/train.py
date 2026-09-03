"""
Q-Knee training CLI: real end-to-end training entrypoint wiring together
ingestion, the PCA/angle-scaling fit, and the VQC training loop, producing
the two artifacts `PipelineRunner` expects at inference time — plus
periodic/best/early-stopped checkpoints along the way.

    DataIngestion (build_dataloaders)
        -> ResNet18FeatureExtractor (frozen)   -> fit QuantumDimReducer -> artifacts/pca_scaler.pkl
        -> QKneeModel (ResNet+PCA+VQC ansatz)   -> training loop        -> artifacts/qknee_model.pt
                                                                         -> artifacts/checkpoints/*.pt

Expects a directory laid out per `qknee.data.dataset.MRIDataset` (ImageFolder
style), pointed to by `--dataset_dir` (default: `config.yaml`'s `paths.data_root`):

    <dataset_dir>/[<plane>/]
        train/
            class_a/*.png|*.jpg|*.npy
            class_b/...
        val/            # optional — if absent, a holdout slice of train/ is used

`--plane` (sagittal/coronal/axial), if given, is joined onto `--dataset_dir`
(`<dataset_dir>/<plane>/train/...`) when that subdirectory exists — the
convention for an MRNet-style dataset root with one ImageFolder-style tree
per anatomical plane; if no such subdirectory exists, `--dataset_dir` is
used as-is (already plane-specific, or plane-agnostic).

Run with:
    python scripts/train.py --dataset_dir data --epochs 50 --learning_rate 0.02
    python scripts/train.py --dataset_dir data --plane sagittal --ansatz data_reuploading
    python scripts/train.py --use_mock --epochs 20          # no real dataset needed
    python scripts/train.py --dry_run                       # 1 batch, 1 epoch, synthetic tensors
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow `python scripts/train.py` to resolve the `qknee` package without
# requiring the caller to set PYTHONPATH or invoke via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import roc_auc_score

from qknee.config.loader import QKneeConfig, load_config, load_config_with_overrides
from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.dataset import build_dataloaders
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.qknee_model import QKneeModel, load_checkpoint, save_checkpoint
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import VQCClassifier
from qknee.models.vqc_data_reuploading import DataReuploadingVQC

logger = get_logger(__name__)

ANSATZ_CHOICES = ("basic", "data_reuploading")
PLANE_CHOICES = ("sagittal", "coronal", "axial")


class TrainingError(RuntimeError):
    """Raised when the training pipeline cannot proceed (bad data layout,
    empty split, device mismatch, etc.) — always carries an actionable message."""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Q-Knee QuantumDimReducer + VQC and persist both artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required-by-spec flags ---
    parser.add_argument("--dataset_dir", type=str, default=None,
                         help="Root directory with train/ (and optionally val/) class subfolders "
                              "(default: config.yaml's paths.data_root). Ignored if --use_mock/--dry_run.")
    parser.add_argument("--plane", choices=PLANE_CHOICES, default=None,
                         help="Anatomical plane subdirectory to train on: <dataset_dir>/<plane>/train/... "
                              "if that subdirectory exists, else --dataset_dir is used as-is.")
    parser.add_argument("--ansatz", choices=ANSATZ_CHOICES, default="basic",
                         help="VQC ansatz: 'basic' (VQCClassifier) or 'data_reuploading' (DataReuploadingVQC).")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Full-batch gradient steps (default: config.yaml's training.n_epochs).")
    parser.add_argument("--batch_size", type=int, default=None,
                         help="DataLoader batch size for ResNet feature extraction "
                              "(default: config.yaml's data.batch_size).")
    parser.add_argument("--learning_rate", type=float, default=None,
                         help="Adam learning rate (default: config.yaml's training.learning_rate).")
    parser.add_argument("--use_mock", action="store_true",
                         help="Train on an in-memory synthetic dataset instead of --dataset_dir — "
                              "useful with no real dataset available (CI, demos). Runs the full "
                              "configured epoch count, unlike --dry_run.")
    parser.add_argument("--dry_run", action="store_true",
                         help="Pipeline-integrity smoke test: 1 batch, 1 epoch, synthetic tensors. "
                              "Overrides --epochs/--use_mock; does not require --dataset_dir.")

    # --- Supporting flags (existing behavior, kept) ---
    parser.add_argument("--pca-artifact", type=str, default=None,
                         help="Output path for the fitted QuantumDimReducer (default: config.yaml's paths.pca_artifact).")
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Output path for the final trained QKneeModel checkpoint "
                              "(default: config.yaml's paths.model_checkpoint).")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                         help="Directory for periodic/best/early-stopped checkpoints during training "
                              "(default: config.yaml's paths.checkpoint_dir).")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pca-fit-max-samples", type=int, default=None,
                         help="Cap on the number of ResNet embeddings collected to fit the PCA reducer.")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Cap on the number of training images used for the full-batch VQC step.")
    parser.add_argument("--val-holdout-fraction", type=float, default=None,
                         help="Fraction of train/ held out for evaluation when no val/ split exists.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                         help="Epochs without validation-loss improvement before stopping early.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None,
                         help="Minimum validation-loss decrease counted as an improvement.")
    parser.add_argument("--device", type=str, default=None,
                         help="Torch device string (default: config.yaml's `device`, or CUDA if available).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible splits/training.")
    parser.add_argument("--plot-loss", action="store_true", default=True,
                         help="Save a loss-curve PNG to config.yaml's paths.eval_output_dir.")
    parser.add_argument("--no-plot-loss", dest="plot_loss", action="store_false")
    return parser


def resolve_config(args: argparse.Namespace) -> QKneeConfig:
    """Builds the nested override dict from every CLI flag the user
    actually supplied, and deep-merges it onto `config.yaml` via
    `qknee.config.loader.load_config_with_overrides` — the "dynamic
    dictionary merging" connection point between the CLI and the config
    file. Flags left at their `None` default don't appear in the override
    dict at all, so they fall through to whatever `config.yaml` (or an
    env-var override) already says.
    """
    overrides: Dict[str, Dict] = {"paths": {}, "data": {}, "training": {}}

    if args.dataset_dir is not None:
        overrides["paths"]["data_root"] = args.dataset_dir
    if args.pca_artifact is not None:
        overrides["paths"]["pca_artifact"] = args.pca_artifact
    if args.checkpoint is not None:
        overrides["paths"]["model_checkpoint"] = args.checkpoint
    if args.checkpoint_dir is not None:
        overrides["paths"]["checkpoint_dir"] = args.checkpoint_dir

    if args.batch_size is not None:
        overrides["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides["data"]["num_workers"] = args.num_workers

    if args.epochs is not None:
        overrides["training"]["n_epochs"] = args.epochs
    if args.learning_rate is not None:
        overrides["training"]["learning_rate"] = args.learning_rate
    if args.pca_fit_max_samples is not None:
        overrides["training"]["pca_fit_max_samples"] = args.pca_fit_max_samples
    if args.max_train_samples is not None:
        overrides["training"]["max_train_samples"] = args.max_train_samples
    if args.val_holdout_fraction is not None:
        overrides["training"]["val_holdout_fraction"] = args.val_holdout_fraction
    if args.early_stopping_patience is not None:
        overrides["training"]["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        overrides["training"]["early_stopping_min_delta"] = args.early_stopping_min_delta

    # Drop empty sections so an all-defaults run is a true no-op merge.
    overrides = {section: values for section, values in overrides.items() if values}

    return load_config_with_overrides(overrides)


def resolve_dataset_dir(dataset_dir: Path, plane: Optional[str]) -> Path:
    """Joins `plane` onto `dataset_dir` (`<dataset_dir>/<plane>`) when that
    subdirectory exists; otherwise returns `dataset_dir` unchanged (already
    plane-specific, or a plane-agnostic layout)."""
    if plane is None:
        return dataset_dir
    plane_dir = dataset_dir / plane
    if plane_dir.is_dir():
        return plane_dir
    logger.warning(
        "--plane=%s given, but %s does not exist; using %s as-is (plane-agnostic layout assumed).",
        plane, plane_dir, dataset_dir,
    )
    return dataset_dir


# --------------------------------------------------------------------------- #
# Ansatz factory
# --------------------------------------------------------------------------- #

def build_vqc(ansatz: str, n_qubits: int, n_layers: int) -> nn.Module:
    """Constructs the requested VQC ansatz. Both share the same
    `(B, n_qubits) -> (B, 1)` sigmoid-probability interface, so either
    plugs directly into `QKneeModel`'s `vqc=` argument."""
    if ansatz == "basic":
        return VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)
    elif ansatz == "data_reuploading":
        return DataReuploadingVQC(n_qubits=n_qubits, n_layers=n_layers)
    raise TrainingError(f"Unknown --ansatz '{ansatz}'; expected one of {ANSATZ_CHOICES}")


# --------------------------------------------------------------------------- #
# Synthetic data (--use_mock / --dry_run)
# --------------------------------------------------------------------------- #

def build_synthetic_image_dataset(
    n_samples: int, image_size: Tuple[int, int], seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic MRI-like (concentric-ring) `(N, 3, H, W)` image tensor +
    `(N,)` binary label tensor — the label weakly shifts the ring radius,
    so there's *some* learnable signal, not pure noise. Used by
    `--use_mock` (full training run, no real dataset required) and
    `--dry_run` (a 1-sample-batch pipeline-integrity check)."""
    rng = np.random.default_rng(seed)
    height, width = image_size
    yy, xx = np.mgrid[0:height, 0:width]
    center_y, center_x = height / 2, width / 2
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    images = np.zeros((n_samples, 3, height, width), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        label = int(rng.integers(0, 2))
        ring_radius = 70 + (15 if label else -15)
        ring = 255 * np.clip(1 - np.abs(radius - ring_radius) / 40, 0, 1)
        texture = rng.normal(scale=8, size=(height, width))
        slice_2d = np.clip(ring + texture, 0, 255).astype(np.float32) / 255.0
        images[i] = np.stack([slice_2d] * 3, axis=0)
        labels[i] = label

    return torch.from_numpy(images), torch.from_numpy(labels)


# --------------------------------------------------------------------------- #
# Stage 1: ResNet feature extraction (drives the PCA fit)
# --------------------------------------------------------------------------- #

def collect_resnet_features(
    loader: torch.utils.data.DataLoader,
    extractor: ResNet18FeatureExtractor,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs the frozen ResNet18 backbone over every batch in `loader` and
    returns `(features, labels)` as `(N, 512)` / `(N,)` numpy arrays."""
    features, labels = [], []
    collected = 0

    extractor.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:  # collate_skip_invalid: every sample in this batch was corrupted
                continue
            images, batch_labels = batch
            images = images.to(device)
            batch_features = extractor(images).cpu().numpy()
            features.append(batch_features)
            labels.append(np.asarray(batch_labels))
            collected += batch_features.shape[0]

            if max_samples is not None and collected >= max_samples:
                break

    if not features:
        raise TrainingError("collect_resnet_features: the provided DataLoader yielded zero batches.")

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    if max_samples is not None:
        features, labels = features[:max_samples], labels[:max_samples]

    return features, labels


# --------------------------------------------------------------------------- #
# Stage 2: full-batch image/label tensors for the training loop
# --------------------------------------------------------------------------- #

def collect_image_tensor(
    loader: torch.utils.data.DataLoader,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenates every batch in `loader` into one `(N, 3, 224, 224)`
    image tensor and one `(N,)` label tensor."""
    images_list, labels_list = [], []
    collected = 0

    for batch in loader:
        if batch is None:  # collate_skip_invalid: every sample in this batch was corrupted
            continue
        images, batch_labels = batch
        images_list.append(images)
        labels_list.append(batch_labels)
        collected += images.shape[0]
        if max_samples is not None and collected >= max_samples:
            break

    if not images_list:
        raise TrainingError("collect_image_tensor: the provided DataLoader yielded zero batches.")

    images_tensor = torch.cat(images_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)
    if max_samples is not None:
        images_tensor, labels_tensor = images_tensor[:max_samples], labels_tensor[:max_samples]

    return images_tensor, labels_tensor


def split_train_holdout(
    images: torch.Tensor, labels: torch.Tensor, holdout_fraction: float, seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Carves a random `holdout_fraction` slice out of `(images, labels)`
    for evaluation, used only when no `val/` directory exists."""
    if not 0.0 < holdout_fraction < 1.0:
        raise TrainingError(f"val_holdout_fraction must be in (0, 1), got {holdout_fraction}")

    n = images.shape[0]
    n_holdout = max(1, int(n * holdout_fraction))
    if n_holdout >= n:
        raise TrainingError(
            f"val_holdout_fraction={holdout_fraction} leaves zero training samples "
            f"out of {n} total — use a smaller fraction or a real val/ split."
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n, generator=generator)
    holdout_idx, train_idx = permutation[:n_holdout], permutation[n_holdout:]

    return images[train_idx], labels[train_idx], images[holdout_idx], labels[holdout_idx]


# --------------------------------------------------------------------------- #
# Training loop: real-time metric logging + checkpointing + early stopping
# --------------------------------------------------------------------------- #

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float]
    val_roc_auc: Optional[float]
    grad_norm: float


def compute_grad_norm(model: nn.Module) -> float:
    """L2 norm of all trainable parameters' gradients combined — a cheap,
    standard signal for spotting exploding/vanishing gradients through the
    quantum layer during training."""
    total_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_sq += float(param.grad.detach().norm(2).item()) ** 2
    return total_sq ** 0.5


def run_training_loop(
    model: QKneeModel,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    eval_images: Optional[torch.Tensor],
    eval_labels: Optional[torch.Tensor],
    n_epochs: int,
    lr: float,
    device: torch.device,
    threshold: float,
    checkpoint_dir: Path,
    log_every: int = 5,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-4,
) -> Tuple[List[EpochMetrics], Optional[Path]]:
    """Full-batch training loop with real-time per-epoch logging (loss,
    accuracy, validation ROC-AUC, gradient norm), periodic + best-model
    checkpointing to `checkpoint_dir`, and early stopping on validation
    loss — guards against the 4-qubit VQC overfitting on the small sample
    sizes typical here (a full-batch gradient step over a few hundred
    images has very little regularizing noise compared to real mini-batch
    SGD, so an unmonitored run can memorize the validation-adjacent split).

    Returns:
        `(history, best_checkpoint_path)` — `best_checkpoint_path` is
        `None` if no validation split was available to select a "best"
        checkpoint from (in which case only periodic checkpoints exist).
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    train_images_dev = train_images.to(device)
    train_labels_dev = train_labels.to(device).float()
    if train_labels_dev.dim() == 1:
        train_labels_dev = train_labels_dev.unsqueeze(1)

    has_eval = eval_images is not None and eval_images.shape[0] > 0

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path: Optional[Path] = None
    history: List[EpochMetrics] = []

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(train_images_dev)
        loss = loss_fn(predictions, train_labels_dev)
        loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()

        train_accuracy = (
            ((predictions.detach() >= threshold).float() == train_labels_dev).float().mean().item()
        )

        val_loss: Optional[float] = None
        val_roc_auc: Optional[float] = None
        if has_eval:
            model.eval()
            with torch.no_grad():
                val_predictions = model(eval_images.to(device))
                val_labels_dev = eval_labels.to(device).float()
                if val_labels_dev.dim() == 1:
                    val_labels_dev = val_labels_dev.unsqueeze(1)
                val_loss = loss_fn(val_predictions, val_labels_dev).item()

                val_probs_np = val_predictions.cpu().numpy().flatten()
                val_labels_np = eval_labels.cpu().numpy().flatten()
                if len(np.unique(val_labels_np)) > 1:
                    val_roc_auc = float(roc_auc_score(val_labels_np, val_probs_np))

        metrics = EpochMetrics(
            epoch=epoch, train_loss=loss.item(), train_accuracy=train_accuracy,
            val_loss=val_loss, val_roc_auc=val_roc_auc, grad_norm=grad_norm,
        )
        history.append(metrics)

        if epoch % max(log_every, 1) == 0 or epoch == n_epochs - 1:
            logger.info(
                "epoch %4d | loss=%.4f | acc=%.4f | val_loss=%s | val_roc_auc=%s | grad_norm=%.4f",
                epoch, metrics.train_loss, metrics.train_accuracy,
                f"{val_loss:.4f}" if val_loss is not None else "n/a",
                f"{val_roc_auc:.4f}" if val_roc_auc is not None else "n/a",
                grad_norm,
            )

        is_last_epoch = epoch == n_epochs - 1
        if (epoch + 1) % max(log_every, 1) == 0 or is_last_epoch:
            periodic_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                model, periodic_path, epoch=epoch,
                extra={"train_loss": metrics.train_loss, "val_loss": val_loss, "val_roc_auc": val_roc_auc},
            )

        if val_loss is not None:
            if val_loss < best_val_loss - early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
                save_checkpoint(
                    model, best_checkpoint_path, epoch=epoch,
                    extra={"val_loss": val_loss, "val_roc_auc": val_roc_auc},
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d: no validation-loss improvement for %d epochs "
                        "(best val_loss=%.4f).", epoch, early_stopping_patience, best_val_loss,
                    )
                    break

    return history, best_checkpoint_path


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_accuracy(
    model: QKneeModel, images: torch.Tensor, labels: torch.Tensor, device: torch.device, threshold: float = 0.5,
) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(images.to(device)).squeeze(1).cpu()
        predicted_labels = (predictions >= threshold).long()
        accuracy = (predicted_labels == labels.long()).float().mean().item()
    return accuracy


def log_history_summary(history: List[EpochMetrics]) -> None:
    if not history:
        logger.warning("Training history is empty — the loop did not run any epochs.")
        return
    losses = [m.train_loss for m in history]
    logger.info(
        "Loss curve: epoch 0 -> %.4f | epoch %d -> %.4f | min %.4f (epoch %d)",
        losses[0], history[-1].epoch, losses[-1], min(losses), int(np.argmin(losses)),
    )


def save_loss_curve_plot(history: List[EpochMetrics], output_dir: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping loss-curve plot.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(13, 4))

    epochs = [m.epoch for m in history]
    ax_loss.plot(epochs, [m.train_loss for m in history], color="#00C896", linewidth=2, label="Train loss")
    val_losses = [m.val_loss for m in history]
    if any(v is not None for v in val_losses):
        ax_loss.plot(epochs, val_losses, color="#E74C3C", linewidth=2, linestyle="--", label="Val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.set_title("Q-Knee VQC Training Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_grad.plot(epochs, [m.grad_norm for m in history], color="#4C8BF5", linewidth=2)
    ax_grad.set_xlabel("Epoch")
    ax_grad.set_ylabel("Gradient Norm (L2)")
    ax_grad.set_title("Gradient Norm")
    ax_grad.grid(alpha=0.3)

    fig.tight_layout()
    output_path = output_dir / "train_loss_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# --------------------------------------------------------------------------- #
# Checkpoint output validation
# --------------------------------------------------------------------------- #

def validate_checkpoint(
    checkpoint_path: Path, reducer: QuantumDimReducer, reference_model: QKneeModel,
    sample_images: torch.Tensor, device: torch.device,
) -> None:
    """Reloads the just-written checkpoint into a fresh `QKneeModel`
    (built with the same ansatz as `reference_model.vqc`) and asserts its
    predictions match `reference_model`'s bit-for-bit."""
    if not checkpoint_path.exists():
        raise TrainingError(f"Checkpoint validation failed: {checkpoint_path} was not written.")

    reloaded_vqc = build_vqc(
        getattr(reference_model.vqc, "_ansatz_name", "basic"),
        reference_model.vqc.n_qubits, reference_model.vqc.n_layers,
    )
    reloaded_model = QKneeModel(
        pca_reducer=reducer, n_qubits=reference_model.vqc.n_qubits, n_layers=reference_model.vqc.n_layers,
        vqc=reloaded_vqc,
    )
    checkpoint = load_checkpoint(reloaded_model, checkpoint_path, map_location=str(device))
    reloaded_model.to(device)
    reloaded_model.eval()

    reference_model.eval()
    with torch.no_grad():
        reference_output = reference_model(sample_images.to(device))
        reloaded_output = reloaded_model(sample_images.to(device))

    torch.testing.assert_close(reference_output, reloaded_output, rtol=1e-5, atol=1e-6)
    logger.info(
        "Checkpoint validation passed: %s round-trips to identical predictions "
        "(epoch=%s, n_qubits=%s, n_layers=%s, size=%.1f KB)",
        checkpoint_path, checkpoint.get("epoch"), checkpoint.get("n_qubits"), checkpoint.get("n_layers"),
        checkpoint_path.stat().st_size / 1024,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    setup_logging()
    args = build_arg_parser().parse_args()

    if args.dry_run:
        args.use_mock = True
        args.epochs = 1

    config = resolve_config(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(
        "=== Q-Knee training run (device=%s, seed=%d, ansatz=%s, mode=%s) ===",
        device, args.seed, args.ansatz,
        "dry_run" if args.dry_run else ("use_mock" if args.use_mock else "dataset"),
    )

    # ------------------------------------------------------------------ #
    # Stage 0: build (or synthesize) train/eval tensors, and a corpus for
    # the PCA fit.
    # ------------------------------------------------------------------ #
    if args.use_mock:
        n_samples = config.data.batch_size if args.dry_run else max(200, config.data.batch_size * 4)
        logger.info("%s: synthesizing %d in-memory images (no --dataset_dir needed).",
                    "Dry run" if args.dry_run else "Mock mode", n_samples)
        all_images, all_labels = build_synthetic_image_dataset(n_samples, config.data.image_size, args.seed)

        if args.dry_run:
            # Literally 1 batch, 1 epoch: no train/eval split, everything
            # (PCA fit corpus, training batch, eval) reuses the same tiny
            # synthetic tensor purely to exercise every stage once.
            train_images, train_labels = all_images, all_labels
            eval_images, eval_labels = all_images, all_labels
        else:
            train_images, train_labels, eval_images, eval_labels = split_train_holdout(
                all_images, all_labels, config.training.val_holdout_fraction, args.seed,
            )
        pca_fit_images = train_images
    else:
        dataset_dir = resolve_dataset_dir(Path(config.paths.data_root), args.plane)
        if not dataset_dir.exists():
            raise TrainingError(
                f"Dataset directory {dataset_dir} does not exist. Expected "
                f"{dataset_dir}/train/<class>/*.png|*.jpg|*.npy (see qknee.data.dataset.MRIDataset), "
                "or pass --use_mock/--dry_run to train without a real dataset."
            )
        logger.info("Training on dataset directory: %s", dataset_dir)

        loaders = build_dataloaders(
            dataset_dir, batch_size=config.data.batch_size, num_workers=config.data.num_workers, labeled=True,
        )
        if "train" not in loaders:
            raise TrainingError(f"No train/ split found under {dataset_dir} — cannot train without labeled data.")
        logger.info("Loaded splits: %s", {name: len(loader.dataset) for name, loader in loaders.items()})

        train_images, train_labels = collect_image_tensor(loaders["train"], max_samples=config.training.max_train_samples)
        logger.info("Collected %d training images.", train_images.shape[0])

        if "val" in loaders:
            eval_images, eval_labels = collect_image_tensor(loaders["val"])
            logger.info("Using %d images from val/ for evaluation.", eval_images.shape[0])
        else:
            logger.warning("No val/ split found — holding out %.0f%% of train/ for evaluation.",
                            config.training.val_holdout_fraction * 100)
            train_images, train_labels, eval_images, eval_labels = split_train_holdout(
                train_images, train_labels, config.training.val_holdout_fraction, args.seed,
            )
            logger.info("Post-holdout: %d train images, %d eval images.", train_images.shape[0], eval_images.shape[0])

        pca_fit_images = train_images[: config.training.pca_fit_max_samples] if config.training.pca_fit_max_samples else train_images

    # ------------------------------------------------------------------ #
    # Stage 1: fit + persist QuantumDimReducer
    # ------------------------------------------------------------------ #
    logger.info("Extracting ResNet18 features to fit QuantumDimReducer (%d images)...", pca_fit_images.shape[0])
    extractor = ResNet18FeatureExtractor(freeze_backbone=config.resnet.freeze_backbone).to(device)
    extractor.eval()
    with torch.no_grad():
        pca_features = extractor(pca_fit_images.to(device)).cpu().numpy()
    logger.info("Collected %d ResNet embeddings (%d-D) for PCA fitting.", *pca_features.shape)

    reducer = QuantumDimReducer(
        use_incremental_pca=config.pca.use_incremental_pca, n_components=config.pca.n_components,
    ).fit(pca_features)
    pca_artifact_path = config.paths.pca_artifact
    pca_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    reducer.save(pca_artifact_path)
    logger.info(
        "Fitted QuantumDimReducer saved to %s (explained variance=%.3f)",
        pca_artifact_path, reducer.explained_variance_ratio_.sum(),
    )

    # ------------------------------------------------------------------ #
    # Stage 2: build QKneeModel with the requested ansatz
    # ------------------------------------------------------------------ #
    vqc = build_vqc(args.ansatz, n_qubits=config.quantum.n_qubits, n_layers=config.quantum.n_layers)
    vqc._ansatz_name = args.ansatz  # stashed for validate_checkpoint's reload
    model = QKneeModel(
        pca_reducer=reducer, n_qubits=config.quantum.n_qubits, n_layers=config.quantum.n_layers,
        freeze_resnet=config.resnet.freeze_backbone, vqc=vqc,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    logger.info(
        "Training QKneeModel: ansatz=%s, %d epoch(s), lr=%g, %d trainable params",
        args.ansatz, config.training.n_epochs, config.training.learning_rate, trainable_params,
    )

    # ------------------------------------------------------------------ #
    # Stage 3: train, with real-time logging + checkpointing + early stopping
    # ------------------------------------------------------------------ #
    checkpoint_dir = config.paths.checkpoint_dir
    start_time = time.perf_counter()
    history, best_checkpoint_path = run_training_loop(
        model, train_images, train_labels, eval_images, eval_labels,
        n_epochs=config.training.n_epochs, lr=config.training.learning_rate, device=device,
        threshold=config.api.tear_risk_threshold, checkpoint_dir=checkpoint_dir,
        log_every=config.training.log_every,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_min_delta=config.training.early_stopping_min_delta,
    )
    elapsed_s = time.perf_counter() - start_time
    log_history_summary(history)
    logger.info("Training loop finished in %.1fs (%d epoch(s) run).", elapsed_s, len(history))

    if args.plot_loss:
        plot_path = save_loss_curve_plot(history, config.paths.eval_output_dir)
        if plot_path is not None:
            logger.info("Saved loss/gradient-norm curves to %s", plot_path)

    # ------------------------------------------------------------------ #
    # Stage 4: evaluate
    # ------------------------------------------------------------------ #
    train_accuracy = evaluate_accuracy(model, train_images, train_labels, device, config.api.tear_risk_threshold)
    eval_accuracy = evaluate_accuracy(model, eval_images, eval_labels, device, config.api.tear_risk_threshold)
    final = history[-1]
    logger.info(
        "Final: train_loss=%.4f | train_acc=%.4f | eval_acc=%.4f | val_loss=%s | val_roc_auc=%s",
        final.train_loss, train_accuracy, eval_accuracy,
        f"{final.val_loss:.4f}" if final.val_loss is not None else "n/a",
        f"{final.val_roc_auc:.4f}" if final.val_roc_auc is not None else "n/a",
    )

    # ------------------------------------------------------------------ #
    # Stage 5: persist + validate the final checkpoint
    # ------------------------------------------------------------------ #
    checkpoint_path = config.paths.model_checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model, checkpoint_path, epoch=final.epoch,
        extra={
            "ansatz": args.ansatz,
            "final_train_loss": final.train_loss,
            "train_accuracy": train_accuracy,
            "eval_accuracy": eval_accuracy,
            "val_roc_auc": final.val_roc_auc,
            "n_train_samples": train_images.shape[0],
            "n_eval_samples": eval_images.shape[0],
        },
    )
    validate_checkpoint(checkpoint_path, reducer, model, eval_images[: min(8, eval_images.shape[0])], device)

    logger.info("=== Training complete ===")
    logger.info("  PCA artifact:       %s", pca_artifact_path.resolve())
    logger.info("  Final checkpoint:   %s", checkpoint_path.resolve())
    logger.info("  Checkpoint dir:     %s", Path(checkpoint_dir).resolve())
    if best_checkpoint_path is not None:
        logger.info("  Best (val_loss) checkpoint: %s", best_checkpoint_path.resolve())
    logger.info("  Eval accuracy: %.4f", eval_accuracy)

    if args.dry_run:
        logger.info("Dry run OK — pipeline integrity verified end-to-end.")


if __name__ == "__main__":
    try:
        main()
    except TrainingError as exc:
        logger.error("Training aborted: %s", exc)
        sys.exit(1)
