"""
CLI entry point: self-supervised (rotation-prediction) pretraining of the
ResNet18 backbone on real, unlabeled RSNA Knee MRI slices — see
qknee/models/ssl_pretrain.py for the pretext-task rationale and
implementation.

Pulls every .dcm slice from train_series/ and the sibling ../rsna-knee/
train_series/ pull (both real Kaggle downloads — see
scripts/_fetch_rsna_pretrain_pool.py), EXCLUDING the 58 studies with full
12-condition label coverage (train.csv) — those are held out for the
labeled PCA->VQC fine-tuning/evaluation pipeline
(scripts/run_ssl_vs_imagenet_kfold.py) and never seen during this
pretraining step, keeping the two stages' data cleanly separated.

Usage:
    python scripts/pretrain_ssl_backbone.py --n-epochs 15 --max-minutes 60
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    from qknee.config.logging_config import setup_logging
    from qknee.data.dataset import RSNA_TARGET_COLUMNS
    from qknee.models.ssl_pretrain import discover_dicom_slices, pretrain_rotation_backbone, save_ssl_backbone

    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-minutes", type=float, default=None, help="Wall-clock training budget.")
    parser.add_argument("--max-slices", type=int, default=None, help="Cap the pretraining pool (debug/smoke runs).")
    parser.add_argument(
        "--checkpoint-path", type=str,
        default=str(REPO_ROOT / "qknee" / "artifacts" / "resnet18_ssl_backbone.pt"),
    )
    args = parser.parse_args()

    train_csv = REPO_ROOT / "train.csv"
    labeled_uids = set()
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        target_cols = [c for c in RSNA_TARGET_COLUMNS if c in train_df.columns]
        labeled_uids = set(
            train_df.loc[train_df[target_cols].notna().any(axis=1), "StudyInstanceUID"].astype(str)
        )
    print(f"[{time.strftime('%H:%M:%S')}] excluding {len(labeled_uids)} labeled study UID(s) from the pretraining pool")

    roots = [REPO_ROOT / "train_series", REPO_ROOT.parent / "rsna-knee" / "train_series"]
    slices = discover_dicom_slices(roots, exclude_study_uids=labeled_uids)
    if args.max_slices is not None:
        slices = slices[: args.max_slices]
    n_studies = len({p.parents[1].name for p in slices})
    print(f"[{time.strftime('%H:%M:%S')}] pretraining pool: {len(slices)} DICOM slice(s) across {n_studies} unlabeled studies")

    if not slices:
        raise SystemExit("No unlabeled DICOM slices found — nothing to pretrain on.")

    max_seconds = args.max_minutes * 60 if args.max_minutes else None
    result = pretrain_rotation_backbone(
        slices,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        max_seconds=max_seconds,
    )

    checkpoint_path = save_ssl_backbone(
        result,
        args.checkpoint_path,
        extra_metadata={
            "n_studies": n_studies,
            "n_labeled_studies_excluded": len(labeled_uids),
            "roots": [str(r) for r in roots],
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
    )
    print(
        f"[{time.strftime('%H:%M:%S')}] DONE: {result.n_epochs_run} epoch(s), "
        f"final rotation-accuracy={result.final_train_accuracy:.4f}, "
        f"backbone saved to {checkpoint_path}"
    )


if __name__ == "__main__":
    main()
