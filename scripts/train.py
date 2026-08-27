"""
Q-Knee training CLI: real end-to-end training entrypoint wiring together
ingestion, the PCA/angle-scaling fit, and the VQC training loop, producing
the two artifacts `PipelineRunner` expects at inference time.

    DataIngestion (build_dataloaders)
        -> ResNet18FeatureExtractor (frozen)   -> fit QuantumDimReducer -> artifacts/pca_scaler.pkl
        -> QKneeModel (ResNet+PCA+VQC)          -> train_qknee_model    -> artifacts/qknee_model.pt

Expects a directory laid out per `qknee.data.dataset.MRIDataset` (ImageFolder
style), pointed to by `config.yaml`'s `paths.data_root`:

    <data_root>/
        train/
            class_a/*.png|*.jpg|*.npy
            class_b/...
        val/            # optional — if absent, a holdout slice of train/ is used
            class_a/...
            class_b/...

Run with:
    python scripts/train.py
    python scripts/train.py --data-root data --epochs 50 --lr 0.02
    python scripts/train.py --device cuda --pca-fit-max-samples 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

# Allow `python scripts/train.py` to resolve the `qknee` package without
# requiring the caller to set PYTHONPATH or invoke via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qknee.config.loader import QKneeConfig, load_config
from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.dataset import build_dataloaders
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.qknee_model import QKneeModel, load_checkpoint, save_checkpoint, train_qknee_model
from qknee.models.resnet_extractor import ResNet18FeatureExtractor

logger = get_logger(__name__)


class TrainingError(RuntimeError):
    """Raised when the training pipeline cannot proceed (bad data layout,
    empty split, device mismatch, etc.) — always carries an actionable message."""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(config: QKneeConfig) -> argparse.Namespace:
    """Defines CLI overrides for every value that's reasonable to sweep from
    the command line; anything omitted falls back to `config.yaml`."""
    parser = argparse.ArgumentParser(
        description="Train the Q-Knee QuantumDimReducer + VQC and persist both artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=str, default=str(config.paths.data_root),
                         help="Root directory with train/ (and optionally val/, test/) class subfolders.")
    parser.add_argument("--pca-artifact", type=str, default=str(config.paths.pca_artifact),
                         help="Output path for the fitted QuantumDimReducer.")
    parser.add_argument("--checkpoint", type=str, default=str(config.paths.model_checkpoint),
                         help="Output path for the trained QKneeModel checkpoint.")
    parser.add_argument("--epochs", type=int, default=config.training.n_epochs,
                         help="Full-batch gradient steps for the VQC training stage.")
    parser.add_argument("--lr", type=float, default=config.training.learning_rate,
                         help="Adam learning rate for the VQC + classical readout.")
    parser.add_argument("--batch-size", type=int, default=config.data.batch_size,
                         help="DataLoader batch size used for ResNet feature extraction.")
    parser.add_argument("--num-workers", type=int, default=config.data.num_workers)
    parser.add_argument("--pca-fit-max-samples", type=int, default=config.training.pca_fit_max_samples,
                         help="Cap on the number of ResNet embeddings collected to fit the PCA reducer.")
    parser.add_argument("--max-train-samples", type=int, default=config.training.max_train_samples,
                         help="Cap on the number of training images used for the full-batch VQC step "
                              "(None = use the entire training split).")
    parser.add_argument("--val-holdout-fraction", type=float, default=config.training.val_holdout_fraction,
                         help="Fraction of train/ held out for evaluation when no val/ split exists.")
    parser.add_argument("--device", type=str, default=None,
                         help="Torch device string (default: config.yaml's `device`, or CUDA if available).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible splits/training.")
    parser.add_argument("--plot-loss", action="store_true", default=True,
                         help="Save a loss-curve PNG to config.yaml's paths.eval_output_dir.")
    parser.add_argument("--no-plot-loss", dest="plot_loss", action="store_false")
    return parser.parse_args()


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
    returns `(features, labels)` as `(N, 512)` / `(N,)` numpy arrays.

    Stops early once `max_samples` embeddings have been collected, so fitting
    the PCA reducer doesn't require a full forward pass over an arbitrarily
    large training set.
    """
    features, labels = [], []
    collected = 0

    extractor.eval()
    with torch.no_grad():
        for images, batch_labels in loader:
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
# Stage 2: full-batch image/label tensors for train_qknee_model
# --------------------------------------------------------------------------- #

def collect_image_tensor(
    loader: torch.utils.data.DataLoader,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenates every batch in `loader` into one `(N, 3, 224, 224)` image
    tensor and one `(N,)` label tensor, for `train_qknee_model`'s full-batch
    gradient-descent API. Capped by `max_samples` to bound memory on large
    training sets."""
    images_list, labels_list = [], []
    collected = 0

    for images, batch_labels in loader:
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
    """Carves a random `holdout_fraction` slice out of `(images, labels)` for
    evaluation, used only when no `val/` directory exists."""
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
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_accuracy(
    model: QKneeModel, images: torch.Tensor, labels: torch.Tensor, device: torch.device, threshold: float = 0.5,
) -> float:
    """Runs `model` in eval mode over `(images, labels)` and returns
    classification accuracy at the given decision threshold."""
    model.eval()
    with torch.no_grad():
        predictions = model(images.to(device)).squeeze(1).cpu()
        predicted_labels = (predictions >= threshold).long()
        accuracy = (predicted_labels == labels.long()).float().mean().item()
    return accuracy


def log_loss_curve(history: list, log_every: int) -> None:
    """Logs a compact summary of the training loss curve — first/last/min
    loss plus periodic checkpoints — without requiring a plotting backend."""
    if not history:
        logger.warning("Loss history is empty — training loop did not run any epochs.")
        return

    logger.info(
        "Loss curve: epoch 0 -> %.4f | epoch %d -> %.4f | min %.4f (epoch %d)",
        history[0], len(history) - 1, history[-1], min(history), int(np.argmin(history)),
    )
    for epoch in range(0, len(history), max(log_every, 1)):
        logger.info("  epoch %4d | loss = %.4f", epoch, history[epoch])
    if (len(history) - 1) % max(log_every, 1) != 0:
        logger.info("  epoch %4d | loss = %.4f", len(history) - 1, history[-1])


def save_loss_curve_plot(history: list, output_dir: Path) -> Optional[Path]:
    """Saves a loss-curve PNG to `output_dir` for visual inspection. Returns
    None (with a warning) instead of raising if matplotlib is unavailable —
    plotting is a nice-to-have, not a training-blocking dependency."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping loss-curve plot.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="#00C896", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("Q-Knee VQC Training Loss")
    ax.grid(alpha=0.3)
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
    """Reloads the just-written checkpoint into a fresh `QKneeModel` and
    asserts its predictions match `reference_model`'s bit-for-bit on
    `sample_images` — catching any save/load round-trip bug (e.g. a state
    dict key mismatch) before the script reports success.
    """
    if not checkpoint_path.exists():
        raise TrainingError(f"Checkpoint validation failed: {checkpoint_path} was not written.")

    reloaded_model = QKneeModel(
        pca_reducer=reducer, n_qubits=reference_model.vqc.n_qubits, n_layers=reference_model.vqc.n_layers,
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
    config = load_config()
    args = parse_args(config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("=== Q-Knee training run (device=%s, seed=%d) ===", device, args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise TrainingError(
            f"--data-root {data_root} does not exist. Expected "
            f"{data_root}/train/<class>/*.png|*.jpg|*.npy (see qknee.data.dataset.MRIDataset)."
        )

    # ------------------------------------------------------------------ #
    # Stage 0: DataLoaders
    # ------------------------------------------------------------------ #
    loaders = build_dataloaders(
        data_root, batch_size=args.batch_size, num_workers=args.num_workers, labeled=True,
    )
    if "train" not in loaders:
        raise TrainingError(f"No train/ split found under {data_root} — cannot train without labeled training data.")
    logger.info("Loaded splits: %s", {name: len(loader.dataset) for name, loader in loaders.items()})

    # ------------------------------------------------------------------ #
    # Stage 1: fit + persist QuantumDimReducer
    # ------------------------------------------------------------------ #
    logger.info("Extracting ResNet18 features to fit QuantumDimReducer (cap=%s samples)...",
                args.pca_fit_max_samples)
    extractor = ResNet18FeatureExtractor(freeze_backbone=config.resnet.freeze_backbone).to(device)
    pca_features, _ = collect_resnet_features(
        loaders["train"], extractor, device, max_samples=args.pca_fit_max_samples,
    )
    logger.info("Collected %d ResNet embeddings (%d-D) for PCA fitting.", *pca_features.shape)

    reducer = QuantumDimReducer(
        use_incremental_pca=config.pca.use_incremental_pca, n_components=config.pca.n_components,
    ).fit(pca_features)
    pca_artifact_path = Path(args.pca_artifact)
    pca_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    reducer.save(pca_artifact_path)
    logger.info(
        "Fitted QuantumDimReducer saved to %s (explained variance=%.3f)",
        pca_artifact_path, reducer.explained_variance_ratio_.sum(),
    )

    # ------------------------------------------------------------------ #
    # Stage 2: build train/eval tensors
    # ------------------------------------------------------------------ #
    train_images, train_labels = collect_image_tensor(loaders["train"], max_samples=args.max_train_samples)
    logger.info("Collected %d training images for the full-batch VQC step.", train_images.shape[0])

    if "val" in loaders:
        eval_images, eval_labels = collect_image_tensor(loaders["val"])
        logger.info("Using %d images from val/ for evaluation.", eval_images.shape[0])
    else:
        logger.warning("No val/ split found — holding out %.0f%% of train/ for evaluation.",
                        args.val_holdout_fraction * 100)
        train_images, train_labels, eval_images, eval_labels = split_train_holdout(
            train_images, train_labels, args.val_holdout_fraction, args.seed,
        )
        logger.info("Post-holdout: %d train images, %d eval images.",
                    train_images.shape[0], eval_images.shape[0])

    # ------------------------------------------------------------------ #
    # Stage 3: train QKneeModel's VQC
    # ------------------------------------------------------------------ #
    model = QKneeModel(
        pca_reducer=reducer, n_qubits=config.quantum.n_qubits, n_layers=config.quantum.n_layers,
        freeze_resnet=config.resnet.freeze_backbone,
    ).to(device)

    logger.info(
        "Training QKneeModel: %d epochs, lr=%g, %d trainable params (VQC weights + readout only)",
        args.epochs, args.lr, sum(p.numel() for p in model.trainable_parameters()),
    )
    history = train_qknee_model(
        model, train_images, train_labels, n_epochs=args.epochs, lr=args.lr,
        device=str(device), log_every=config.training.log_every,
    )
    log_loss_curve(history, config.training.log_every)

    if args.plot_loss:
        plot_path = save_loss_curve_plot(history, config.paths.eval_output_dir)
        if plot_path is not None:
            logger.info("Saved loss-curve plot to %s", plot_path)

    # ------------------------------------------------------------------ #
    # Stage 4: evaluate
    # ------------------------------------------------------------------ #
    train_accuracy = evaluate_accuracy(model, train_images, train_labels, device, config.api.tear_risk_threshold)
    eval_accuracy = evaluate_accuracy(model, eval_images, eval_labels, device, config.api.tear_risk_threshold)
    logger.info("Final train accuracy: %.4f | eval accuracy: %.4f (threshold=%.2f)",
                train_accuracy, eval_accuracy, config.api.tear_risk_threshold)

    # ------------------------------------------------------------------ #
    # Stage 5: persist + validate checkpoint
    # ------------------------------------------------------------------ #
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model, checkpoint_path, epoch=args.epochs,
        extra={
            "final_train_loss": history[-1],
            "train_accuracy": train_accuracy,
            "eval_accuracy": eval_accuracy,
            "n_train_samples": train_images.shape[0],
            "n_eval_samples": eval_images.shape[0],
        },
    )
    validate_checkpoint(checkpoint_path, reducer, model, eval_images[: min(8, eval_images.shape[0])], device)

    logger.info("=== Training complete ===")
    logger.info("  PCA artifact: %s", pca_artifact_path.resolve())
    logger.info("  Checkpoint:   %s", checkpoint_path.resolve())
    logger.info("  Eval accuracy: %.4f", eval_accuracy)


if __name__ == "__main__":
    try:
        main()
    except TrainingError as exc:
        logger.error("Training aborted: %s", exc)
        sys.exit(1)
