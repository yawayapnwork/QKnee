"""
Effusion-specific: does augmenting each training fold with weak-labeled
studies (qknee/artifacts/effusion_scaled_labels.csv, the 3,723 "confidently"
classifier-labeled studies with no `needs_review` flag) improve out-of-fold
macro-AUC over the 58-ground-truth-only baseline
(qknee/artifacts/ssl_vs_imagenet_kfold_summary_seed*.json, `imagenet_only`
backbone, `per_condition_auc["Effusion"]`)?

Provenance discipline (never merge indistinguishable):
    - every row used carries an explicit `provenance` tag,
      "ground_truth" (one of the 58) or "weak_labeled" (one of the 3,723
      classifier-labeled studies with images available on disk).
    - folds are built ONLY from the 58 ground-truth studies, with the EXACT
      same `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`
      split on Effusion labels as the baseline run
      (scripts/run_ssl_vs_imagenet_kfold.py's run_oof_macro_auc_analysis),
      so held-out evaluation is on ground-truth studies only, every seed.
    - for each of the 5 folds, the TRAINING side only is augmented with
      weak-labeled studies, chosen to match the training fold's existing
      Effusion class balance (positive:negative ratio) rather than
      whatever ratio the weak-labeled pool happens to have -- capped by
      whichever class has fewer available weak-labeled images.
    - the held-out fold (X_test/y_test) is untouched ground truth, same as
      every other reported result.

Requires a `qknee/artifacts/effusion_expanded_manifest.csv` provenance
manifest (build it first via scripts/build_effusion_expanded_manifest.py)
and weak-labeled DICOM images already downloaded into train_series/ (see
scripts/_fetch_rsna_effusion_weak_labeled.py) -- studies whose images are
not yet on disk are simply unavailable augmentation candidates, not errors.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_SERIES_DIR = REPO_ROOT.parent / "rsna-knee" / "train_series"
WEAK_SERIES_DIR = REPO_ROOT.parent / "rsna-knee" / "train_series"
LABELS_CSV = REPO_ROOT / "qknee" / "artifacts" / "effusion_scaled_labels.csv"
GT_CACHE = REPO_ROOT / "qknee" / "artifacts" / "rsna58_features_imagenet.npz"
WEAK_CACHE = REPO_ROOT / "qknee" / "artifacts" / "effusion_weak_labeled_features_imagenet.npz"
CONDITION = "Effusion"
CONDITION_IDX = 7  # position of "Effusion" in qknee.data.dataset.RSNA_TARGET_COLUMNS -- must match
                    # run_ssl_vs_imagenet_kfold.py's condition_names ordering for init_seed parity


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_ground_truth_features() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Reuses the existing 58-study ImageNet-backbone feature cache (same
    cache the baseline run consumed) so ground-truth features/labels/order
    are bit-identical to the baseline -- no re-extraction, no drift."""
    if not GT_CACHE.exists():
        raise FileNotFoundError(
            f"{GT_CACHE} not found -- run scripts/run_ssl_vs_imagenet_kfold.py "
            "at least once first to build the ground-truth feature cache."
        )
    data = np.load(GT_CACHE, allow_pickle=False)
    features = data["features"]
    labels = data[f"label_{CONDITION}"]
    uids = [str(u) for u in data["uids"]]
    return features, labels, uids


def load_weak_labeled_candidates() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV, dtype={"StudyInstanceUID": str})
    confident = df[df["needs_review"].isna() | (df["needs_review"].astype(str).str.strip() == "")].copy()
    return confident[["StudyInstanceUID", "tier", "assigned_label"]].reset_index(drop=True)


def _study_has_images(study_uid: str) -> bool:
    study_dir = WEAK_SERIES_DIR / study_uid
    if not study_dir.is_dir():
        return False
    return any(study_dir.rglob("*.dcm"))


def build_or_load_weak_features(
    candidate_uids: List[str], resnet_extractor,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Extracts (and caches, keyed by UID) one 512-D ResNet18 embedding per
    weak-labeled study whose images are currently on disk. Returns
    `(uid -> feature)` and `uid -> assigned_label` for every study that
    was embeddable; studies without images (or with unreadable DICOMs) are
    silently absent from the returned dicts -- not an error, since the
    weak-labeled image pool is still being downloaded incrementally."""
    from qknee.data.dataset import RSNAKneeDataset
    from qknee.data.ingestion import DataIngestion

    cached_features: Dict[str, np.ndarray] = {}
    if WEAK_CACHE.exists():
        data = np.load(WEAK_CACHE, allow_pickle=False)
        cached_uids = [str(u) for u in data["uids"]]
        for i, u in enumerate(cached_uids):
            cached_features[u] = data["features"][i]
        log(f"loaded {len(cached_features)} cached weak-labeled feature(s) from {WEAK_CACHE}")

    to_extract = [u for u in candidate_uids if u not in cached_features and _study_has_images(u)]
    log(f"{len(to_extract)} weak-labeled studies have images on disk and need extraction "
        f"({len(candidate_uids) - len(to_extract) - sum(1 for u in candidate_uids if u in cached_features)} "
        "still missing images)")

    if to_extract:
        labels_df = pd.DataFrame({"StudyInstanceUID": to_extract, "Effusion": 0.0})
        tmp_csv = REPO_ROOT / "qknee" / "artifacts" / "_tmp_weak_labeled_extract.csv"
        labels_df.to_csv(tmp_csv, index=False)
        try:
            dataset = RSNAKneeDataset(
                tmp_csv, WEAK_SERIES_DIR,
                series_csv_path=REPO_ROOT / "train_series.csv",
                require_targets=False,
            )
            resnet_extractor.eval()
            ingestion = DataIngestion(train=False)
            with torch.no_grad():
                for record in dataset:
                    series_dirs = [d for dirs in record.plane_series_dirs.values() for d in dirs]
                    if not series_dirs:
                        continue
                    series_features = []
                    for series_path in series_dirs:
                        try:
                            batch = ingestion.preprocess(series_path)
                            feature_vector = resnet_extractor(batch)
                        except Exception as exc:  # noqa: BLE001
                            log(f"  study {record.study_instance_uid}: series {series_path} failed: {exc}")
                            continue
                        series_features.append(feature_vector.squeeze(0).numpy())
                    if not series_features:
                        continue
                    cached_features[record.study_instance_uid] = np.mean(series_features, axis=0).astype(np.float32)
        finally:
            tmp_csv.unlink(missing_ok=True)

        WEAK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        uids_out = list(cached_features.keys())
        feats_out = np.stack([cached_features[u] for u in uids_out]).astype(np.float32)
        np.savez(WEAK_CACHE, features=feats_out, uids=np.array(uids_out))
        log(f"cached {len(uids_out)} total weak-labeled feature(s) -> {WEAK_CACHE}")

    return cached_features


def train_vqc(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    n_epochs: int, lr: float, n_qubits: int, n_layers: int, init_seed: int,
) -> np.ndarray:
    from qknee.models.pca_reducer import QuantumDimReducer
    from qknee.models.vqc import VQCClassifier

    reducer = QuantumDimReducer(use_incremental_pca=False)
    q_train = reducer.fit_transform(X_train)
    q_test = reducer.transform(X_test)

    torch.manual_seed(init_seed)
    model = VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.from_numpy(q_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_t = torch.from_numpy(q_test).float()

    final_test_probs = None
    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            final_test_probs = model(X_test_t).squeeze(1).numpy()
    return final_test_probs


def select_balanced_augmentation(
    weak_pos_uids: List[str], weak_neg_uids: List[str], fold_pos: int, fold_neg: int, rng: np.random.RandomState,
) -> Tuple[List[str], List[str]]:
    """Picks weak-labeled UIDs to add to a training fold so the ADDED
    pool's positive:negative ratio matches the fold's own ratio
    (`fold_pos`:`fold_neg`), capped by whichever class has fewer available
    weak-labeled candidates. Uses ALL available candidates of the
    limiting class and downsamples the other class to match the ratio,
    rather than an arbitrary fixed augmentation size."""
    if fold_pos == 0 or fold_neg == 0 or not weak_pos_uids or not weak_neg_uids:
        return [], []

    ratio = fold_pos / fold_neg  # pos per neg
    # max pos we could add if we use ALL available neg:
    max_pos_from_all_neg = len(weak_neg_uids) * ratio
    # max neg we could add if we use ALL available pos:
    max_neg_from_all_pos = len(weak_pos_uids) / ratio

    if max_pos_from_all_neg <= len(weak_pos_uids):
        n_pos = int(round(max_pos_from_all_neg))
        n_neg = len(weak_neg_uids)
    else:
        n_pos = len(weak_pos_uids)
        n_neg = int(round(max_neg_from_all_pos))

    n_pos = min(n_pos, len(weak_pos_uids))
    n_neg = min(n_neg, len(weak_neg_uids))

    chosen_pos = list(rng.choice(weak_pos_uids, size=n_pos, replace=False)) if n_pos > 0 else []
    chosen_neg = list(rng.choice(weak_neg_uids, size=n_neg, replace=False)) if n_neg > 0 else []
    return chosen_pos, chosen_neg


def run_seed(
    seed: int, gt_features: np.ndarray, gt_labels: np.ndarray, gt_uids: List[str],
    weak_features_by_uid: Dict[str, np.ndarray], weak_labels_by_uid: Dict[str, int],
    n_folds: int, n_epochs: int, lr: float, n_qubits: int, n_layers: int,
) -> Dict:
    n_samples = len(gt_uids)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_probs = np.full(n_samples, np.nan, dtype=np.float64)

    weak_pos_all = [u for u, y in weak_labels_by_uid.items() if y == 1]
    weak_neg_all = [u for u, y in weak_labels_by_uid.items() if y == 0]
    rng = np.random.RandomState(seed)

    fold_details = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(gt_features, gt_labels)):
        X_train_gt, X_test = gt_features[train_idx], gt_features[test_idx]
        y_train_gt, y_test = gt_labels[train_idx], gt_labels[test_idx]

        fold_pos = int((y_train_gt == 1).sum())
        fold_neg = int((y_train_gt == 0).sum())
        aug_pos_uids, aug_neg_uids = select_balanced_augmentation(
            weak_pos_all, weak_neg_all, fold_pos, fold_neg, rng,
        )
        aug_uids = aug_pos_uids + aug_neg_uids
        if aug_uids:
            X_aug = np.stack([weak_features_by_uid[u] for u in aug_uids]).astype(np.float32)
            y_aug = np.array([weak_labels_by_uid[u] for u in aug_uids], dtype=np.int64)
            X_train = np.concatenate([X_train_gt, X_aug], axis=0)
            y_train = np.concatenate([y_train_gt, y_aug], axis=0)
        else:
            X_train, y_train = X_train_gt, y_train_gt

        init_seed = seed * 1_000_000 + CONDITION_IDX * 1_000 + fold_idx
        test_probs = train_vqc(X_train, y_train, X_test, y_test, n_epochs, lr, n_qubits, n_layers, init_seed)
        oof_probs[test_idx] = test_probs

        fold_details.append({
            "fold": fold_idx,
            "gt_train_pos": fold_pos, "gt_train_neg": fold_neg,
            "n_weak_added_pos": len(aug_pos_uids), "n_weak_added_neg": len(aug_neg_uids),
            "n_train_total": int(len(y_train)), "n_test": int(len(y_test)),
        })

    oof_auc = float(roc_auc_score(gt_labels, oof_probs)) if len(np.unique(gt_labels)) == 2 else None
    return {"seed": seed, "oof_macro_auc": oof_auc, "fold_details": fold_details}


def main() -> None:
    from qknee.config.logging_config import setup_logging
    from qknee.models.resnet_extractor import ResNet18FeatureExtractor

    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--macro-n-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument(
        "--output", type=str,
        default=str(REPO_ROOT / "qknee" / "artifacts" / "effusion_weak_augmented_kfold_summary.json"),
    )
    args = parser.parse_args()

    gt_features, gt_labels, gt_uids = load_ground_truth_features()
    log(f"ground-truth: {len(gt_uids)} studies, {int((gt_labels==1).sum())} positive / "
        f"{int((gt_labels==0).sum())} negative (Effusion)")

    candidates = load_weak_labeled_candidates()
    log(f"weak-labeled candidate pool (confident, no needs_review): {len(candidates)} studies")

    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    weak_features_by_uid = build_or_load_weak_features(candidates["StudyInstanceUID"].tolist(), extractor)

    weak_labels_by_uid = {
        row.StudyInstanceUID: int(row.assigned_label)
        for row in candidates.itertuples()
        if row.StudyInstanceUID in weak_features_by_uid
    }
    n_weak_pos = sum(1 for v in weak_labels_by_uid.values() if v == 1)
    n_weak_neg = sum(1 for v in weak_labels_by_uid.values() if v == 0)
    log(f"weak-labeled studies usable (image on disk + extracted): {len(weak_labels_by_uid)} "
        f"({n_weak_pos} positive / {n_weak_neg} negative) out of {len(candidates)} confident candidates")

    per_seed_results = []
    for seed in args.seeds:
        log(f"=== seed {seed} ===")
        result = run_seed(
            seed, gt_features, gt_labels, gt_uids, weak_features_by_uid, weak_labels_by_uid,
            args.n_folds, args.macro_n_epochs, args.lr, args.n_qubits, args.n_layers,
        )
        log(f"  -> oof_macro_auc (Effusion, weak-augmented) = {result['oof_macro_auc']}")
        per_seed_results.append(result)

    aucs = [r["oof_macro_auc"] for r in per_seed_results if r["oof_macro_auc"] is not None]
    summary = {
        "condition": CONDITION,
        "n_ground_truth_studies": len(gt_uids),
        "n_weak_labeled_candidates_confident": len(candidates),
        "n_weak_labeled_usable_with_images": len(weak_labels_by_uid),
        "n_weak_labeled_usable_pos": n_weak_pos,
        "n_weak_labeled_usable_neg": n_weak_neg,
        "per_seed_results": per_seed_results,
        "oof_macro_auc_spread": {
            "mean": float(np.mean(aucs)) if aucs else None,
            "min": float(np.min(aucs)) if aucs else None,
            "max": float(np.max(aucs)) if aucs else None,
            "std": float(np.std(aucs)) if aucs else None,
            "n_seeds": len(aucs),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {output_path}")
    print(json.dumps(summary["oof_macro_auc_spread"], indent=2))


if __name__ == "__main__":
    main()
