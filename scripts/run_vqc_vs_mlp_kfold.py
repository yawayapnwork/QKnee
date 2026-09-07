"""
Compares the PCA->VQC pipeline against a parameter-matched classical MLP
head — same PCA(4)-reduced ImageNet-ResNet18 features, same seeded 5-fold
CV, same "no early stopping, track every epoch" curve protocol — on the
same 58 fully-labeled RSNA Knee studies used by
scripts/run_ssl_vs_imagenet_kfold.py.

The question this answers: does the 4-qubit variational circuit itself buy
anything over a classical model with the *same number of trainable
parameters* (41: `VQCClassifier(n_qubits=4, n_layers=3)`'s 36 quantum
rotation weights + 5 in its `Linear(4, 1)` readout), or is the VQC's
observed performance indistinguishable from what any 41-parameter
classical model would get on 58 samples?

`MatchedMLP` (Linear(4, 8, bias=False) -> ReLU -> Linear(8, 1, bias=True))
has exactly 32 + 8 + 1 = 41 trainable parameters, matching the VQC exactly
(verified at import time by `_assert_param_counts_match`).

Reuses the same per-fold/per-condition seeding fix as
run_ssl_vs_imagenet_kfold.py: `torch.manual_seed(init_seed)` immediately
before constructing either model, with `init_seed` deterministically
derived from `--seed`, so weight init is controlled and reproducible
instead of depending on whatever state the global RNG was left in.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELED_SERIES_DIR = REPO_ROOT.parent / "rsna-knee" / "train_series"


class MatchedMLP(nn.Module):
    """Classical head with the same trainable-parameter count as
    `VQCClassifier(n_qubits=4, n_layers=3)` (41), same (B, 4) -> (B, 1)
    interface, same Sigmoid output range."""

    def __init__(self, n_qubits: int = 4, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(n_qubits, hidden, bias=False)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc2(self.act(self.fc1(x))))


def _assert_param_counts_match() -> None:
    from qknee.models.vqc import VQCClassifier

    vqc_params = sum(p.numel() for p in VQCClassifier(n_qubits=4, n_layers=3).parameters())
    mlp_params = sum(p.numel() for p in MatchedMLP(n_qubits=4, hidden=8).parameters())
    if vqc_params != mlp_params:
        raise AssertionError(
            f"MatchedMLP has {mlp_params} trainable params, VQCClassifier has "
            f"{vqc_params} -- update MatchedMLP's `hidden` so the comparison stays "
            f"parameter-matched."
        )


def build_model(model_name: str, n_qubits: int, n_layers: int) -> nn.Module:
    if model_name == "vqc":
        from qknee.models.vqc import VQCClassifier

        return VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)
    if model_name == "mlp_matched":
        return MatchedMLP(n_qubits=n_qubits, hidden=8)
    raise ValueError(f"unknown model_name: {model_name!r}")


def train_with_curve(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    model_name: str, n_epochs: int, lr: float, n_qubits: int, n_layers: int,
    init_seed: int,
) -> Tuple[List[float], List[float], np.ndarray]:
    """Same training/eval protocol as `run_ssl_vs_imagenet_kfold.train_vqc_with_curve`
    (full-batch Adam, BCE loss, AUC tracked every epoch, no early stopping),
    generalized to either model. `init_seed` seeds torch immediately before
    model construction -- see module docstring."""

    def safe_auc(y_true, y_prob) -> float:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))

    torch.manual_seed(init_seed)
    model = build_model(model_name, n_qubits=n_qubits, n_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test_t = torch.from_numpy(X_test).float()

    train_auc_curve: List[float] = []
    test_auc_curve: List[float] = []
    final_test_probs = None

    for _epoch in range(n_epochs):
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
    model_name: str, n_folds: int, n_epochs: int, lr: float, n_qubits: int, n_layers: int,
    seed: int,
) -> Dict:
    y = labels[condition]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_train_curves: List[List[float]] = []
    fold_test_curves: List[List[float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        train_curve, test_curve, _ = train_with_curve(
            X_train, y_train, X_test, y_test, model_name=model_name, n_epochs=n_epochs, lr=lr,
            n_qubits=n_qubits, n_layers=n_layers, init_seed=seed * 1_000 + fold_idx,
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
    model_name: str, n_folds: int, n_epochs: int, lr: float, n_qubits: int, n_layers: int,
    seed: int,
) -> Dict:
    from qknee.models.evaluate import compute_macro_auc

    n_samples = features.shape[0]
    y_true_full: Dict[str, np.ndarray] = {}
    y_prob_full: Dict[str, np.ndarray] = {}

    for condition_idx, condition in enumerate(condition_names):
        y = labels[condition]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_probs = np.full(n_samples, np.nan, dtype=np.float64)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            _, _, final_test_probs = train_with_curve(
                X_train, y_train, X_test, y_test, model_name=model_name, n_epochs=n_epochs, lr=lr,
                n_qubits=n_qubits, n_layers=n_layers,
                init_seed=seed * 1_000_000 + condition_idx * 1_000 + fold_idx,
            )
            oof_probs[test_idx] = final_test_probs

        y_true_full[condition] = y
        y_prob_full[condition] = oof_probs
        print(f"    [{condition}] out-of-fold AUC = {roc_auc_score(y, oof_probs):.4f}")

    return compute_macro_auc(y_true_full, y_prob_full, condition_names=condition_names)


def _load_features(cache_path: Path, condition_names: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Loads a pre-built real-feature cache (as written by
    `run_ssl_vs_imagenet_kfold.build_or_load_features`). This script never
    extracts features itself -- it's a head-architecture comparison, not a
    backbone comparison, so it requires the ImageNet-backbone cache to
    already exist."""
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found -- run scripts/run_ssl_vs_imagenet_kfold.py "
            f"first (with any --ssl-checkpoint) to build the ImageNet feature cache, "
            f"or point --features-cache at an existing rsna58_features_*.npz."
        )
    data = np.load(cache_path, allow_pickle=False)
    features = data["features"]
    labels = {c: data[f"label_{c}"] for c in condition_names}
    print(f"[{time.strftime('%H:%M:%S')}] loaded cached features from {cache_path} ({features.shape})")
    return features, labels


def main() -> None:
    from qknee.config.logging_config import setup_logging
    from qknee.data.dataset import RSNA_TARGET_COLUMNS
    from qknee.models.pca_reducer import QuantumDimReducer

    setup_logging()
    _assert_param_counts_match()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--features-cache", type=str,
        default=str(REPO_ROOT / "qknee" / "artifacts" / "rsna58_features_imagenet.npz"),
        help="Pre-built (N, 512) real ResNet18 feature cache (see build_or_load_features "
             "in run_ssl_vs_imagenet_kfold.py). Not re-extracted here.",
    )
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
        default=str(REPO_ROOT / "qknee" / "artifacts" / "vqc_vs_mlp_kfold_summary.json"),
    )
    args = parser.parse_args()

    condition_names = list(RSNA_TARGET_COLUMNS)

    results: Dict[str, Dict] = {}
    for model_name in ("vqc", "mlp_matched"):
        print(f"\n=== model: {model_name} ===")
        features, labels = _load_features(Path(args.features_cache), condition_names)

        print(f"[{time.strftime('%H:%M:%S')}] per-epoch curve analysis on '{args.curve_condition}' "
              f"({args.n_folds}-fold, {args.curve_n_epochs}-epoch ceiling)...")

        # Fold-wise PCA(4)->[0, 2pi] reduction, fit on the fold's train split
        # only (no test-set leakage), exactly mirroring
        # run_ssl_vs_imagenet_kfold.train_vqc_with_curve's per-fold QuantumDimReducer.
        def reduced_curve_analysis() -> Dict:
            y = labels[args.curve_condition]
            skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
            fold_train_curves: List[List[float]] = []
            fold_test_curves: List[List[float]] = []
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
                reducer = QuantumDimReducer(use_incremental_pca=False)
                q_train = reducer.fit_transform(features[train_idx])
                q_test = reducer.transform(features[test_idx])
                y_train, y_test = y[train_idx], y[test_idx]
                train_curve, test_curve, _ = train_with_curve(
                    q_train, y_train, q_test, y_test, model_name=model_name,
                    n_epochs=args.curve_n_epochs, lr=args.lr, n_qubits=args.n_qubits,
                    n_layers=args.n_layers, init_seed=args.seed * 1_000 + fold_idx,
                )
                fold_train_curves.append(train_curve)
                fold_test_curves.append(test_curve)
                print(
                    f"    fold {fold_idx + 1}/{args.n_folds}: final train_auc={train_curve[-1]:.4f}, "
                    f"final test_auc={test_curve[-1]:.4f}, peak test_auc={np.nanmax(test_curve):.4f} "
                    f"@ epoch {int(np.nanargmax(test_curve))}"
                )
            mean_train_curve = np.nanmean(np.array(fold_train_curves), axis=0)
            mean_test_curve = np.nanmean(np.array(fold_test_curves), axis=0)
            peak_epoch = int(np.nanargmax(mean_test_curve))
            return {
                "condition": args.curve_condition,
                "n_folds": args.n_folds,
                "n_epochs": args.curve_n_epochs,
                "fold_train_auc_curves": fold_train_curves,
                "fold_test_auc_curves": fold_test_curves,
                "mean_train_auc_curve": mean_train_curve.tolist(),
                "mean_test_auc_curve": mean_test_curve.tolist(),
                "peak_mean_test_auc": float(mean_test_curve[peak_epoch]),
                "peak_epoch": peak_epoch,
                "final_mean_test_auc": float(mean_test_curve[-1]),
                "final_mean_train_auc": float(mean_train_curve[-1]),
            }

        curve_result = reduced_curve_analysis()
        print(
            f"  -> peak mean test AUC {curve_result['peak_mean_test_auc']:.4f} @ epoch "
            f"{curve_result['peak_epoch']}; final (epoch {args.curve_n_epochs - 1}) mean test AUC "
            f"{curve_result['final_mean_test_auc']:.4f}"
        )

        print(f"[{time.strftime('%H:%M:%S')}] out-of-fold macro-AUC across all {len(condition_names)} "
              f"conditions ({args.n_folds}-fold, {args.macro_n_epochs} epochs/fold)...")

        def reduced_oof_macro_auc_analysis() -> Dict:
            from qknee.models.evaluate import compute_macro_auc

            n_samples = features.shape[0]
            y_true_full: Dict[str, np.ndarray] = {}
            y_prob_full: Dict[str, np.ndarray] = {}
            for condition_idx, condition in enumerate(condition_names):
                y = labels[condition]
                skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
                oof_probs = np.full(n_samples, np.nan, dtype=np.float64)
                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, y)):
                    reducer = QuantumDimReducer(use_incremental_pca=False)
                    q_train = reducer.fit_transform(features[train_idx])
                    q_test = reducer.transform(features[test_idx])
                    y_train, y_test = y[train_idx], y[test_idx]
                    _, _, final_test_probs = train_with_curve(
                        q_train, y_train, q_test, y_test, model_name=model_name,
                        n_epochs=args.macro_n_epochs, lr=args.lr, n_qubits=args.n_qubits,
                        n_layers=args.n_layers,
                        init_seed=args.seed * 1_000_000 + condition_idx * 1_000 + fold_idx,
                    )
                    oof_probs[test_idx] = final_test_probs
                y_true_full[condition] = y
                y_prob_full[condition] = oof_probs
                print(f"    [{condition}] out-of-fold AUC = {roc_auc_score(y, oof_probs):.4f}")
            return compute_macro_auc(y_true_full, y_prob_full, condition_names=condition_names)

        macro_result = reduced_oof_macro_auc_analysis()
        print(f"  -> macro-AUC (out-of-fold, {len(condition_names)}-condition) = {macro_result['final_score']:.4f}")

        results[model_name] = {
            "n_studies": features.shape[0],
            "epoch_curve": curve_result,
            "oof_macro_auc": macro_result,
        }

    vqc_curve = results["vqc"]["epoch_curve"]
    mlp_curve = results["mlp_matched"]["epoch_curve"]
    verdict = {
        "curve_condition": args.curve_condition,
        "vqc_peak_epoch": vqc_curve["peak_epoch"],
        "mlp_matched_peak_epoch": mlp_curve["peak_epoch"],
        "vqc_peak_test_auc": vqc_curve["peak_mean_test_auc"],
        "mlp_matched_peak_test_auc": mlp_curve["peak_mean_test_auc"],
        "vqc_oof_macro_auc": results["vqc"]["oof_macro_auc"]["final_score"],
        "mlp_matched_oof_macro_auc": results["mlp_matched"]["oof_macro_auc"]["final_score"],
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
