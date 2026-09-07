"""
Compares the PCA->VQC pipeline's real-data behavior under two ResNet18
backbones — plain ImageNet-pretrained (the pipeline's current default) vs.
further self-supervised (rotation-prediction) pretrained on real, unlabeled
RSNA knee MRI slices (qknee/models/ssl_pretrain.py) — on the same 58
fully-labeled RSNA Knee studies, with 5-fold cross-validation.

Two questions this answers:

  1. Overfitting pattern: does per-epoch TEST-set AUC still peak in the
     first ~40 epochs and then degrade (the earlier diagnosis on this
     58-sample dataset), or does a backbone that has actually seen however
     many real knee images the pretraining pool held before ever touching
     the 58 labeled ones hold up longer before overfitting? Tracked
     epoch-by-epoch (no early stopping) for one representative, reasonably
     class-balanced condition (--curve-condition, default ACL: 24/34
     positive/negative on the full 58) across all 5 folds, then averaged
     fold-wise.

  2. Overall performance: does the SSL-pretrained backbone's macro-AUC
     (RSNA's official metric — mean AUC across all 12 conditions) beat,
     match, or lag the ImageNet-only backbone's, using proper out-of-fold
     evaluation (every one of the 58 studies is scored by the fold model
     that did NOT train on it, then one macro-AUC is computed over all 58
     aggregated predictions per condition — avoids the degenerate
     single-class-in-a-tiny-test-fold AUC problem a single 75/25 split or
     naive per-fold AUC averaging would hit on some of the rarer
     conditions, e.g. MCL at 9/58 positive).

Requires a pretrained SSL backbone checkpoint from
scripts/pretrain_ssl_backbone.py (--ssl-checkpoint).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELED_SERIES_DIR = REPO_ROOT.parent / "rsna-knee" / "train_series"


def _checkpoint_fingerprint(checkpoint_path: Optional[Path]) -> str:
    """Cheap identity for the backbone weights a feature cache was built
    from: `None` (plain ImageNet, which never changes at runtime) gets a
    fixed sentinel; a real checkpoint path is fingerprinted by its
    resolved path + size + mtime, so re-pretraining the SSL backbone (or
    pointing at a different checkpoint) changes the fingerprint without
    needing to hash the (tens-of-MB) file contents."""
    if checkpoint_path is None:
        return "imagenet_only:no_checkpoint"
    stat = Path(checkpoint_path).stat()
    return f"{Path(checkpoint_path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def build_or_load_features(
    cache_path: Path, resnet_extractor, condition_names, checkpoint_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    from qknee.models.evaluate import build_rsna_feature_dataset

    fingerprint = _checkpoint_fingerprint(checkpoint_path)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        cached_fingerprint = str(data["checkpoint_fingerprint"][0]) if "checkpoint_fingerprint" in data else None
        if cached_fingerprint == fingerprint:
            features = data["features"]
            uids = list(data["uids"])
            labels = {c: data[f"label_{c}"] for c in condition_names}
            print(f"[{time.strftime('%H:%M:%S')}] loaded cached features from {cache_path} ({features.shape})")
            return features, labels, uids
        print(
            f"[{time.strftime('%H:%M:%S')}] cache at {cache_path} was built from a different backbone "
            f"checkpoint (cached={cached_fingerprint!r}, current={fingerprint!r}) -- treating as stale, re-extracting"
        )

    print(f"[{time.strftime('%H:%M:%S')}] extracting real RSNA features -> {cache_path} (no cache found)")
    features, labels, uids = build_rsna_feature_dataset(
        REPO_ROOT / "train.csv", DEFAULT_LABELED_SERIES_DIR, resnet_extractor=resnet_extractor,
        condition_names=condition_names,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {f"label_{c}": labels[c] for c in condition_names}
    np.savez(
        cache_path, features=features, uids=np.array(uids),
        checkpoint_fingerprint=np.array([fingerprint]), **save_kwargs,
    )
    return features, labels, uids


def train_vqc_with_curve(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    n_epochs: int, lr: float, n_qubits: int, n_layers: int, use_incremental_pca: bool,
) -> Tuple[List[float], List[float], np.ndarray]:
    """Trains one PCA(n_qubits)->VQC model for a fixed `n_epochs` ceiling
    (no early stopping — the whole point is to see the un-truncated curve),
    computing train- and test-set ROC-AUC after every epoch.

    Returns:
        `(train_auc_per_epoch, test_auc_per_epoch, final_test_probs)` — an
        AUC entry is `float('nan')` for an epoch where the relevant split
        has only one class present (can happen on tiny folds), same
        "exclude, don't fabricate" contract as `compute_macro_auc`.
    """
    from qknee.models.pca_reducer import QuantumDimReducer
    from qknee.models.vqc import VQCClassifier

    reducer = QuantumDimReducer(use_incremental_pca=use_incremental_pca)
    q_train = reducer.fit_transform(X_train)
    q_test = reducer.transform(X_test)

    model = VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.from_numpy(q_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_t = torch.from_numpy(q_test).float()

    def safe_auc(y_true, y_prob) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))

    train_auc_curve: List[float] = []
    test_auc_curve: List[float] = []
    final_test_probs = None

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        train_probs_t = model(X_train_t)
        loss = loss_fn(train_probs_t, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_probs = model(X_train_t).squeeze(1).numpy()
            test_probs = model(X_test_t).squeeze(1).numpy()
        train_auc_curve.append(safe_auc(y_train, train_probs))
        test_auc_curve.append(safe_auc(y_test, test_probs))
        final_test_probs = test_probs

    return train_auc_curve, test_auc_curve, final_test_probs


def run_epoch_curve_analysis(
    features: np.ndarray, labels: Dict[str, np.ndarray], condition: str,
    n_folds: int, n_epochs: int, lr: float, n_qubits: int, n_layers: int,
    use_incremental_pca: bool, seed: int,
) -> Dict:
    y = labels[condition]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_train_curves: List[List[float]] = []
    fold_test_curves: List[List[float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_curve, test_curve, _ = train_vqc_with_curve(
            X_train, y_train, X_test, y_test, n_epochs=n_epochs, lr=lr,
            n_qubits=n_qubits, n_layers=n_layers, use_incremental_pca=use_incremental_pca,
        )
        fold_train_curves.append(train_curve)
        fold_test_curves.append(test_curve)
        print(
            f"    fold {fold_idx + 1}/{n_folds}: final train_auc={train_curve[-1]:.4f}, "
            f"final test_auc={test_curve[-1]:.4f}, peak test_auc={np.nanmax(test_curve):.4f} "
            f"@ epoch {int(np.nanargmax(test_curve))}"
        )

    mean_train_curve = np.nanmean(np.array(fold_train_curves), axis=0)
    mean_test_curve = np.nanmean(np.array(fold_test_curves), axis=0)
    peak_epoch = int(np.nanargmax(mean_test_curve))

    return {
        "condition": condition,
        "n_folds": n_folds,
        "n_epochs": n_epochs,
        "fold_train_auc_curves": fold_train_curves,
        "fold_test_auc_curves": fold_test_curves,
        "mean_train_auc_curve": mean_train_curve.tolist(),
        "mean_test_auc_curve": mean_test_curve.tolist(),
        "peak_mean_test_auc": float(mean_test_curve[peak_epoch]),
        "peak_epoch": peak_epoch,
        "final_mean_test_auc": float(mean_test_curve[-1]),
        "final_mean_train_auc": float(mean_train_curve[-1]),
    }


def run_oof_macro_auc_analysis(
    features: np.ndarray, labels: Dict[str, np.ndarray], condition_names,
    n_folds: int, n_epochs: int, lr: float, n_qubits: int, n_layers: int,
    use_incremental_pca: bool, seed: int,
) -> Dict:
    from qknee.models.evaluate import compute_macro_auc

    n_samples = features.shape[0]
    y_true_full: Dict[str, np.ndarray] = {}
    y_prob_full: Dict[str, np.ndarray] = {}

    for condition in condition_names:
        y = labels[condition]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_probs = np.full(n_samples, np.nan, dtype=np.float64)

        for train_idx, test_idx in skf.split(features, y):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            _, _, final_test_probs = train_vqc_with_curve(
                X_train, y_train, X_test, y_test, n_epochs=n_epochs, lr=lr,
                n_qubits=n_qubits, n_layers=n_layers, use_incremental_pca=use_incremental_pca,
            )
            oof_probs[test_idx] = final_test_probs

        y_true_full[condition] = y
        y_prob_full[condition] = oof_probs
        print(f"    [{condition}] out-of-fold AUC = {roc_auc_score(y, oof_probs):.4f}")

    return compute_macro_auc(y_true_full, y_prob_full, condition_names=condition_names)


def main() -> None:
    from qknee.config.logging_config import setup_logging
    from qknee.data.dataset import RSNA_TARGET_COLUMNS
    from qknee.models.resnet_extractor import ResNet18FeatureExtractor

    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssl-checkpoint", type=str, required=True)
    parser.add_argument("--curve-condition", type=str, default="ACL")
    parser.add_argument("--curve-n-epochs", type=int, default=200)
    parser.add_argument("--macro-n-epochs", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=str,
        default=str(REPO_ROOT / "qknee" / "artifacts" / "ssl_vs_imagenet_kfold_summary.json"),
    )
    args = parser.parse_args()

    condition_names = list(RSNA_TARGET_COLUMNS)

    backbones = {
        "imagenet_only": {
            "extractor": ResNet18FeatureExtractor(freeze_backbone=True),
            "cache": REPO_ROOT / "qknee" / "artifacts" / "rsna58_features_imagenet.npz",
            "checkpoint_path": None,
        },
        "ssl_pretrained": {
            "extractor": ResNet18FeatureExtractor(
                freeze_backbone=True, pretrained_backbone_path=args.ssl_checkpoint,
            ),
            "cache": REPO_ROOT / "qknee" / "artifacts" / "rsna58_features_ssl.npz",
            "checkpoint_path": Path(args.ssl_checkpoint),
        },
    }

    results: Dict[str, Dict] = {}
    for backbone_name, spec in backbones.items():
        print(f"\n=== backbone: {backbone_name} ===")
        features, labels, uids = build_or_load_features(
            spec["cache"], spec["extractor"], condition_names, checkpoint_path=spec["checkpoint_path"],
        )

        print(f"[{time.strftime('%H:%M:%S')}] per-epoch curve analysis on '{args.curve_condition}' "
              f"({args.n_folds}-fold, {args.curve_n_epochs}-epoch ceiling)...")
        curve_result = run_epoch_curve_analysis(
            features, labels, args.curve_condition, args.n_folds, args.curve_n_epochs,
            args.lr, args.n_qubits, args.n_layers, use_incremental_pca=False, seed=args.seed,
        )
        print(
            f"  -> peak mean test AUC {curve_result['peak_mean_test_auc']:.4f} @ epoch "
            f"{curve_result['peak_epoch']}; final (epoch {args.curve_n_epochs - 1}) mean test AUC "
            f"{curve_result['final_mean_test_auc']:.4f}"
        )

        print(f"[{time.strftime('%H:%M:%S')}] out-of-fold macro-AUC across all {len(condition_names)} "
              f"conditions ({args.n_folds}-fold, {args.macro_n_epochs} epochs/fold)...")
        macro_result = run_oof_macro_auc_analysis(
            features, labels, condition_names, args.n_folds, args.macro_n_epochs,
            args.lr, args.n_qubits, args.n_layers, use_incremental_pca=False, seed=args.seed,
        )
        print(f"  -> macro-AUC (out-of-fold, {len(condition_names)}-condition) = {macro_result['final_score']:.4f}")

        results[backbone_name] = {
            "n_studies": features.shape[0],
            "epoch_curve": curve_result,
            "oof_macro_auc": macro_result,
        }

    imagenet_curve = results["imagenet_only"]["epoch_curve"]
    ssl_curve = results["ssl_pretrained"]["epoch_curve"]
    verdict = {
        "curve_condition": args.curve_condition,
        "imagenet_only_peak_epoch": imagenet_curve["peak_epoch"],
        "ssl_pretrained_peak_epoch": ssl_curve["peak_epoch"],
        "imagenet_only_peak_test_auc": imagenet_curve["peak_mean_test_auc"],
        "ssl_pretrained_peak_test_auc": ssl_curve["peak_mean_test_auc"],
        "overfitting_pattern_changed": ssl_curve["peak_epoch"] > imagenet_curve["peak_epoch"] + 5,
        "imagenet_only_oof_macro_auc": results["imagenet_only"]["oof_macro_auc"]["final_score"],
        "ssl_pretrained_oof_macro_auc": results["ssl_pretrained"]["oof_macro_auc"]["final_score"],
    }
    results["verdict"] = verdict

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nFull results -> {output_path}")


if __name__ == "__main__":
    main()
