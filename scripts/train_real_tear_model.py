"""
Trains a REAL ACL/meniscal tear-risk VQC head on this project's actual
n=58 RSNA Knee ground-truth studies (the `ACL`, `Medial Meniscus`, and
`Lateral Meniscus` columns of `train.csv`) -- and overwrites
`qknee/artifacts/qknee_model.pt` / `qknee/artifacts/pca_scaler.pkl`, the
exact paths `config.yaml`'s `paths.model_checkpoint`/`paths.pca_artifact`
already point to, so `extras/api/server.py`'s `/predict` serves it with no
further wiring changes.

Why this script exists: the checkpoint previously at that path was
byte-identical to `qknee/artifacts/diagnostic_effusion_full/qknee_model.pt`
-- trained on Effusion labels -- while `/predict` labels every result
"Tear Detected"/"Normal" and the whole product frames itself as ACL/
meniscal tear triage. That model had never seen a tear label. This script
trains one that actually has, on the same real ground truth
`RESULTS.md`/`README.md` already cite (n=58, the real labeled ceiling in
this competition dataset). The Effusion-trained original is untouched at
`qknee/artifacts/diagnostic_effusion_full/` (already a full, separate
backup) if it's ever needed again.

Binary target: 1 if ACL == 1 OR Medial Meniscus == 1 OR Lateral Meniscus
== 1 (matches the product's existing "Tear Detected"/"Normal" framing and
the README's "ACL & meniscal tear risk triage" claim); 0 only when all
three are present and 0. A study missing all three columns is skipped.

Mirrors `qknee.data.dataset.RSNAEffusionTrainDataset` and
`scripts/train.py --use_rsna_effusion`'s pipeline stage-for-stage (same
`DataIngestion` decode path, same `ResNet18FeatureExtractor` -> `PCA` fit
-> `QKneeModel` training loop, same checkpoint format) -- just pointed at
a different label column and the real (not weak-labeled/expanded)
58-study ground-truth pool, since no equivalent expanded pool exists for
ACL/Meniscus.

Usage:
    python scripts/train_real_tear_model.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.dataset import RSNAKneeDataset, collate_skip_invalid
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.qknee_model import QKneeModel, save_checkpoint
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import VQCClassifier
from scripts.build_real_demo_cases import _pick_series_dir
from scripts.train import (
    TrainingError,
    collect_image_tensor,
    evaluate_accuracy,
    log_history_summary,
    run_training_loop,
    split_train_holdout,
    validate_checkpoint,
)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = REPO_ROOT / "train.csv"
TRAIN_SERIES_CSV = REPO_ROOT / "train_series.csv"
TRAIN_SERIES_DIR = REPO_ROOT / "train_series"
TEAR_LABEL_COLUMNS = ("ACL", "Medial Meniscus", "Lateral Meniscus")


class RSNATearTrainDataset(Dataset):
    """`torch.utils.data.Dataset` adapter -- same shape/contract as
    `RSNAEffusionTrainDataset` -- pairing each of the real 58 ground-truth
    RSNA Knee studies with a combined ACL-or-meniscus-tear binary label
    instead of the Effusion column."""

    def __init__(self) -> None:
        from qknee.data.ingestion import DataIngestion

        self._ingestion = DataIngestion(train=False)
        self._rsna_dataset = RSNAKneeDataset(
            TRAIN_CSV, TRAIN_SERIES_DIR, series_csv_path=TRAIN_SERIES_CSV, require_targets=True,
        )

    @staticmethod
    def _normalize_uint8(slice_2d):
        import numpy as np

        slice_2d = slice_2d.astype("float32")
        min_val, max_val = float(slice_2d.min()), float(slice_2d.max())
        if max_val > min_val:
            slice_2d = (slice_2d - min_val) / (max_val - min_val)
        else:
            slice_2d = np.zeros_like(slice_2d)
        return (slice_2d * 255).astype("uint8")

    def __len__(self) -> int:
        return len(self._rsna_dataset)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, float]]:
        record = self._rsna_dataset[index]
        if record.targets is None:
            return None
        raw_values = [record.targets.get(col) for col in TEAR_LABEL_COLUMNS]
        if all(v is None for v in raw_values):
            return None
        tear_label = 1.0 if any((v is not None and v >= 0.5) for v in raw_values) else 0.0

        study_dir = TRAIN_SERIES_DIR / record.study_instance_uid
        if not study_dir.is_dir():
            return None
        series_dir = _pick_series_dir(study_dir)
        if series_dir is None:
            return None

        try:
            volume = self._ingestion.load_dicom_series(series_dir)
            primary_slice_index = volume.shape[0] // 2 if volume.ndim == 3 else 0
            central_slice_raw = volume[primary_slice_index] if volume.ndim == 3 else volume
            display_slice = self._normalize_uint8(central_slice_raw)
            batch = self._ingestion.preprocess(display_slice)  # (1, 1, 3, 224, 224)
        except Exception as exc:  # noqa: BLE001 - one bad study must not abort the whole epoch
            logger.warning(
                "RSNATearTrainDataset: study %s series %s failed to decode (%s); skipping.",
                record.study_instance_uid, series_dir, exc,
            )
            return None

        tensor = batch.squeeze(0).squeeze(0)  # (3, 224, 224)
        return tensor, tear_label


def main() -> None:
    setup_logging()
    from qknee.config.loader import load_config

    config = load_config()
    seed = 0
    torch.manual_seed(seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("=== Real ACL/meniscal tear-risk training run (device=%s) ===", device)
    logger.info("Label columns combined (OR) into a single tear target: %s", TEAR_LABEL_COLUMNS)

    dataset = RSNATearTrainDataset()
    loader = DataLoader(
        dataset, batch_size=config.data.batch_size, num_workers=0, shuffle=False,
        collate_fn=collate_skip_invalid,
    )
    all_images, all_labels = collect_image_tensor(loader)
    n_pos = int(all_labels.sum().item())
    logger.info(
        "Collected %d real, ground-truth-labeled tear images (%d positive / %d negative).",
        all_images.shape[0], n_pos, all_images.shape[0] - n_pos,
    )
    if n_pos == 0 or n_pos == all_images.shape[0]:
        raise TrainingError(
            f"Collected labels are single-class ({n_pos}/{all_images.shape[0]} positive) -- "
            "cannot train a binary classifier. Aborting rather than producing a checkpoint that "
            "always predicts one class."
        )

    train_images, train_labels, eval_images, eval_labels = split_train_holdout(
        all_images, all_labels, config.training.val_holdout_fraction, seed,
    )
    logger.info("Post-holdout: %d train images, %d eval images.", train_images.shape[0], eval_images.shape[0])

    # ------------------------------------------------------------------ #
    # Fit + persist QuantumDimReducer (PCA) on this real tear-training pool
    # -- overwrites the Effusion-pool-fit PCA at the same config path so
    # the two artifacts (PCA + VQC) stay a matched pair, exactly as
    # scripts/train.py's own invariant requires.
    # ------------------------------------------------------------------ #
    extractor = ResNet18FeatureExtractor(freeze_backbone=config.resnet.freeze_backbone).to(device)
    extractor.eval()
    with torch.no_grad():
        pca_features = extractor(train_images.to(device)).cpu().numpy()
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
    # Build + train QKneeModel (angle-encoding VQC ansatz, matching the
    # checkpoint format the production `/predict` path already expects)
    # ------------------------------------------------------------------ #
    vqc = VQCClassifier(n_qubits=config.quantum.n_qubits, n_layers=config.quantum.n_layers, ansatz="angle")
    # `"basic"` -- not `"angle"` -- matches `scripts/train.py`'s own CLI-name
    # convention: `build_vqc("basic", ...)` is what constructs this exact
    # `VQCClassifier(ansatz="angle")`. `validate_checkpoint` below reloads
    # via `build_vqc(reference_model.vqc._ansatz_name, ...)`, so this must
    # be a name `build_vqc` actually accepts, not the lower-level `ansatz=`
    # value passed to `VQCClassifier` itself.
    vqc._ansatz_name = "basic"
    model = QKneeModel(
        pca_reducer=reducer, n_qubits=config.quantum.n_qubits, n_layers=config.quantum.n_layers,
        freeze_resnet=config.resnet.freeze_backbone, vqc=vqc,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    logger.info(
        "Training QKneeModel: %d epoch(s), lr=%g, %d trainable params",
        config.training.n_epochs, config.training.learning_rate, trainable_params,
    )

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

    train_accuracy = evaluate_accuracy(model, train_images, train_labels, device, config.api.tear_risk_threshold)
    eval_accuracy = evaluate_accuracy(model, eval_images, eval_labels, device, config.api.tear_risk_threshold)
    final = history[-1]
    logger.info(
        "Final: train_loss=%.4f | train_acc=%.4f | eval_acc=%.4f | val_loss=%s | val_roc_auc=%s",
        final.train_loss, train_accuracy, eval_accuracy,
        f"{final.val_loss:.4f}" if final.val_loss is not None else "n/a",
        f"{final.val_roc_auc:.4f}" if final.val_roc_auc is not None else "n/a",
    )

    checkpoint_path = config.paths.model_checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model, checkpoint_path, epoch=final.epoch,
        extra={
            "ansatz": "angle",
            "target": "acl_or_meniscus_tear (ACL | Medial Meniscus | Lateral Meniscus, real RSNA ground truth, n=58)",
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
    logger.info("  PCA artifact:     %s", pca_artifact_path.resolve())
    logger.info("  Final checkpoint: %s", checkpoint_path.resolve())
    logger.info("  Eval accuracy:    %.4f", eval_accuracy)


if __name__ == "__main__":
    try:
        main()
    except TrainingError as exc:
        logger.error("Training aborted: %s", exc)
        sys.exit(1)
