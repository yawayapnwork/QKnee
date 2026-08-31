"""
Unified end-to-end Q-Knee model: Image -> ResNet18 (512-D) -> PCA (4-D) ->
4-qubit PennyLane VQC -> Sigmoid risk probability, expressed as a single
`nn.Module` (`QKneeModel`) plus training-loop and checkpoint utilities.

Why the PCA stage is a torch layer, not sklearn:
    `quantum_dim_reduction.QuantumDimReducer` fits StandardScaler -> PCA(4)
    -> MinMaxScaler using scikit-learn. Each of those three stages is an
    affine transform, so once fitted they can be losslessly re-expressed as
    plain tensor arithmetic. `PCAProjectionLayer.from_reducer(...)` does
    exactly that conversion, producing a frozen (non-trainable) `nn.Module`
    that reproduces the fitted reducer's `transform()` output bit-for-bit,
    but as ordinary differentiable torch ops with no sklearn/numpy
    dependency at inference time and no autograd-graph break between the
    CNN and the quantum layer.

`QKneeModel` composes:
    1. `ResNet18FeatureExtractor` (frozen backbone, from qknee/models/resnet_extractor.py)
    2. `PCAProjectionLayer`       (frozen, built from a pre-fitted QuantumDimReducer)
    3. `VQCClassifier`            (trainable quantum layer + sigmoid readout,
                                   from qknee/models/vqc.py)

Only the VQC's quantum-circuit weights and its classical readout layer are
trainable — the ResNet backbone and PCA projection are both frozen, so
`train_qknee_model` naturally trains only those parameters via
`filter(lambda p: p.requires_grad, model.parameters())`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger
from qknee.data.dataset import RSNA_TARGET_COLUMNS
from qknee.models.pca_reducer import ANGLE_RANGE, QuantumDimReducer
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import N_QUBITS, VQCClassifier
from qknee.models.vqc_multitarget import MultiTargetHeadType, build_multi_target_head

logger = get_logger(__name__)
_config = load_config()


class PCAProjectionLayer(nn.Module):
    """Frozen torch re-implementation of a fitted
    `QuantumDimReducer` (StandardScaler -> PCA -> MinMaxScaler), so the
    512-D -> 4-D reduction is plain differentiable tensor arithmetic instead
    of an autograd-breaking call out to sklearn.

    All parameters are registered as buffers (not `nn.Parameter`), so they
    move with the module across devices/dtypes via `.to(...)` but are never
    updated by an optimizer.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("std_mean", torch.zeros(in_features))
        self.register_buffer("std_scale", torch.ones(in_features))
        self.register_buffer("pca_mean", torch.zeros(in_features))
        self.register_buffer("pca_components", torch.zeros(out_features, in_features))
        self.register_buffer("minmax_scale", torch.ones(out_features))
        self.register_buffer("minmax_min", torch.zeros(out_features))

    @classmethod
    def from_reducer(cls, reducer: QuantumDimReducer) -> "PCAProjectionLayer":
        """Builds a `PCAProjectionLayer` whose buffers reproduce
        `reducer.transform(x)` exactly, from an already-fitted
        `QuantumDimReducer`."""
        if not reducer._is_fitted:
            raise ValueError("Cannot build a PCAProjectionLayer from an unfitted QuantumDimReducer")

        in_features = reducer.standard_scaler.mean_.shape[0]
        out_features = reducer.n_components
        layer = cls(in_features=in_features, out_features=out_features)

        layer.std_mean.copy_(torch.from_numpy(reducer.standard_scaler.mean_).float())
        layer.std_scale.copy_(torch.from_numpy(reducer.standard_scaler.scale_).float())
        layer.pca_mean.copy_(torch.from_numpy(reducer.pca.mean_).float())
        layer.pca_components.copy_(torch.from_numpy(reducer.pca.components_).float())
        # sklearn MinMaxScaler exposes precomputed `scale_`/`min_` such that
        # transform(x) == x * scale_ + min_ (feature_range already baked in).
        layer.minmax_scale.copy_(torch.from_numpy(reducer.minmax_scaler.scale_).float())
        layer.minmax_min.copy_(torch.from_numpy(reducer.minmax_scaler.min_).float())

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {x.shape[-1]}")

        standardized = (x - self.std_mean) / self.std_scale
        projected = (standardized - self.pca_mean) @ self.pca_components.T
        angles = projected * self.minmax_scale + self.minmax_min
        return torch.clamp(angles, min=ANGLE_RANGE[0], max=ANGLE_RANGE[1])


class QKneeModel(nn.Module):
    """Unified Q-Knee model: Image/Tensor -> ResNet18 (512-D) -> PCA (4-D)
    -> 4-qubit VQC -> Sigmoid risk probability.

    Args:
        pca_reducer: A pre-fitted `QuantumDimReducer` (see
            quantum_dim_reduction.py). Required — the PCA stage cannot be
            trained jointly (it must be fit on a representative corpus of
            512-D ResNet features beforehand).
        n_qubits: Number of qubits in the VQC (must equal `pca_reducer.n_components`).
        n_layers: Variational circuit depth.
        freeze_resnet: Whether to freeze the ResNet18 backbone (default True;
            this is the standard "frozen feature extractor" setup used
            throughout this project).
    """

    def __init__(
        self,
        pca_reducer: QuantumDimReducer,
        n_qubits: int = N_QUBITS,
        n_layers: int = _config.quantum.n_layers,
        freeze_resnet: bool = _config.resnet.freeze_backbone,
        vqc: Optional[nn.Module] = None,
    ):
        super().__init__()

        if pca_reducer.n_components != n_qubits:
            raise ValueError(
                f"pca_reducer produces {pca_reducer.n_components}-D output but "
                f"n_qubits={n_qubits}; these must match."
            )

        self.resnet = ResNet18FeatureExtractor(freeze_backbone=freeze_resnet)
        self.pca_layer = PCAProjectionLayer.from_reducer(pca_reducer)
        # `vqc` lets a caller swap in a different ansatz (e.g.
        # `DataReuploadingVQC`, `StronglyEntanglingVQCClassifier` — see
        # `scripts/train.py --ansatz`) as long as it exposes the same
        # `(B, n_qubits) -> (B, 1)` sigmoid-probability interface
        # `VQCClassifier` does; defaults to the standard `VQCClassifier`.
        self.vqc = vqc if vqc is not None else VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)

        # PCA layer is a fixed, deterministic re-expression of a fitted
        # sklearn pipeline — never trainable, regardless of freeze_resnet.
        for param in self.pca_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) single-slice batch, or (B, S, 3, 224, 224)
               multi-slice volume batch (averaged into one embedding per
               volume by the ResNet stage — see resnet_feature_extractor.py).

        Returns:
            (B, 1) tensor of risk probabilities in [0, 1].
        """
        features_512d = self.resnet(x)          # (B, 512)
        quantum_angles = self.pca_layer(features_512d)  # (B, n_qubits), in [0, 2*pi]
        risk_probability = self.vqc(quantum_angles)      # (B, 1), in [0, 1]
        return risk_probability

    def trainable_parameters(self):
        """Returns only the parameters that should be handed to an
        optimizer (the VQC's quantum weights + classical readout — the
        ResNet backbone and PCA projection are both frozen)."""
        return filter(lambda p: p.requires_grad, self.parameters())


class QKneeMultiTargetModel(nn.Module):
    """Multi-label variant of `QKneeModel`, for the 12 RSNA Knee
    conditions: Image/Tensor -> ResNet18 (512-D) -> PCA (4-D) -> a
    multi-target quantum-classical head (`qknee.models.vqc_multitarget`)
    -> 12 raw logits, in `qknee.data.dataset.RSNA_TARGET_COLUMNS` order.

    IMPORTANT — unlike `QKneeModel` (whose `forward()` returns calibrated
    `[0, 1]` probabilities directly, since it trains with plain `BCELoss`
    on one target), this model's `forward()` returns RAW LOGITS. That's
    the correct input for `nn.BCEWithLogitsLoss`
    (`train_qknee_multitarget_model` below) — its numerically-fused
    sigmoid+BCE computation is more stable under the severe per-condition
    class imbalance RSNA has (e.g. Fracture vs. Effusion prevalence) than
    training against already-squashed probabilities would be. Call
    `.predict_proba(x)` for calibrated `[0.0, 1.0]` sigmoid probabilities
    — e.g. for `scripts/generate_kaggle_submission.py`'s `submission.csv`.

    Args:
        pca_reducer: A pre-fitted `QuantumDimReducer` (same requirement as
            `QKneeModel`).
        n_qubits: Qubits per quantum sub-circuit (must equal
            `pca_reducer.n_components`).
        n_layers: Variational circuit depth.
        freeze_resnet: Whether to freeze the ResNet18 backbone.
        head_type: `"multi_observable"` or `"ensemble"` (see
            `qknee.models.vqc_multitarget`); defaults to
            `config.quantum.multi_target_head`. Ignored if `head` is given.
        head: A pre-built `(B, n_qubits) -> (B, 12)` raw-logits module,
            overriding `head_type`'s default construction — for a caller
            that wants to inject a custom/pre-trained head directly.
    """

    TARGET_NAMES: Sequence[str] = RSNA_TARGET_COLUMNS

    def __init__(
        self,
        pca_reducer: QuantumDimReducer,
        n_qubits: int = N_QUBITS,
        n_layers: int = _config.quantum.n_layers,
        freeze_resnet: bool = _config.resnet.freeze_backbone,
        head_type: Optional[MultiTargetHeadType] = None,
        head: Optional[nn.Module] = None,
    ):
        super().__init__()

        if pca_reducer.n_components != n_qubits:
            raise ValueError(
                f"pca_reducer produces {pca_reducer.n_components}-D output but "
                f"n_qubits={n_qubits}; these must match."
            )

        self.resnet = ResNet18FeatureExtractor(freeze_backbone=freeze_resnet)
        self.pca_layer = PCAProjectionLayer.from_reducer(pca_reducer)
        self.head_type: MultiTargetHeadType = head_type or _config.quantum.multi_target_head
        self.head = head if head is not None else build_multi_target_head(
            self.head_type, n_qubits=n_qubits, n_layers=n_layers,
        )

        for param in self.pca_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: `(B, 3, 224, 224)` single-slice batch, or `(B, S, 3, 224, 224)`
               multi-slice volume batch.

        Returns:
            `(B, 12)` RAW LOGITS (pre-sigmoid) — see class docstring. Use
            `.predict_proba(x)` for calibrated `[0, 1]` probabilities.
        """
        features_512d = self.resnet(x)                  # (B, 512)
        quantum_angles = self.pca_layer(features_512d)   # (B, n_qubits), in [0, 2*pi]
        logits = self.head(quantum_angles)                # (B, 12), raw logits
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Calibrated per-condition probabilities: `torch.sigmoid(self(x))`,
        in `[0.0, 1.0]` — matching the Kaggle `submission.csv` schema
        (`qknee.data.dataset.RSNA_TARGET_COLUMNS` column order)."""
        return torch.sigmoid(self.forward(x))

    def trainable_parameters(self):
        """Returns only the parameters that should be handed to an
        optimizer (the multi-target head's quantum weights + any classical
        projection layers — the ResNet backbone and PCA projection are
        both frozen)."""
        return filter(lambda p: p.requires_grad, self.parameters())


# --------------------------------------------------------------------------- #
# Training utility
# --------------------------------------------------------------------------- #

def train_qknee_model(
    model: QKneeModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = _config.training.n_epochs,
    lr: float = _config.training.learning_rate,
    device: Optional[str] = None,
    log_every: int = _config.training.log_every,
) -> List[float]:
    """Trains `model`'s trainable parameters (VQC weights + readout) with
    Binary Cross-Entropy loss and the Adam optimizer.

    Args:
        model: A `QKneeModel` instance.
        inputs: (N, 3, 224, 224) or (N, S, 3, 224, 224) training batch.
        labels: (N,) or (N, 1) binary labels (0 = no tear, 1 = tear).
        n_epochs: Number of full-batch gradient steps.
        lr: Adam learning rate.
        device: Optional device string ("cpu"/"cuda"); defaults to CUDA if available.
        log_every: Print loss every this many epochs.

    Returns:
        List of per-epoch loss values (for plotting/inspection).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()

    inputs = inputs.to(device)
    labels = labels.to(device).float()
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    history: List[float] = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            logger.info("epoch %3d | loss = %.4f", epoch, loss.item())

    return history


# --------------------------------------------------------------------------- #
# Multi-target training utility (RSNA 12-condition head)
# --------------------------------------------------------------------------- #

def compute_pos_weight(labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Computes per-class positive weights for `nn.BCEWithLogitsLoss`
    from a `(N, 12)` multi-hot label tensor — the standard class-imbalance
    correction, `pos_weight_c = n_negative_c / n_positive_c`: a rare
    positive class (e.g. Fracture, typically far less prevalent than
    Effusion in RSNA-style knee MRI data) gets a large `pos_weight`, so a
    false negative on it costs the loss as much as `pos_weight` false
    negatives on a perfectly-balanced class would — preventing the model
    from minimizing loss by simply always predicting "negative" for rare
    conditions.

    Args:
        labels: `(N, 12)` binary (0/1) tensor.
        eps: Floor applied to both the positive count (before dividing)
            and the final weight, so a condition with zero positives in
            this batch/split doesn't produce a `inf`/`NaN` weight.

    Returns:
        `(12,)` tensor of per-class positive weights, `>= eps`.
    """
    n_samples = labels.shape[0]
    n_positive = labels.sum(dim=0).clamp(min=eps)
    n_negative = n_samples - labels.sum(dim=0)
    return (n_negative / n_positive).clamp(min=eps)


def train_qknee_multitarget_model(
    model: QKneeMultiTargetModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = _config.training.n_epochs,
    lr: float = _config.training.learning_rate,
    device: Optional[str] = None,
    log_every: int = _config.training.log_every,
    pos_weight: Optional[torch.Tensor] = None,
) -> List[float]:
    """Trains `model`'s trainable parameters (multi-target head weights)
    with `nn.BCEWithLogitsLoss` (raw-logit input — see
    `QKneeMultiTargetModel`'s docstring for why not plain `BCELoss`) and
    the Adam optimizer.

    Args:
        model: A `QKneeMultiTargetModel` instance.
        inputs: `(N, 3, 224, 224)` or `(N, S, 3, 224, 224)` training batch.
        labels: `(N, 12)` multi-hot binary labels, in
            `QKneeMultiTargetModel.TARGET_NAMES` order.
        n_epochs: Number of full-batch gradient steps.
        lr: Adam learning rate.
        device: Optional device string; defaults to CUDA if available.
        log_every: Log loss every this many epochs.
        pos_weight: Optional `(12,)` per-class positive weight tensor for
            `BCEWithLogitsLoss`; computed from `labels` via
            `compute_pos_weight` when omitted (the common case — a caller
            with a separately-computed, dataset-wide `pos_weight` should
            pass it explicitly instead of letting it be re-derived from
            just this training batch).

    Returns:
        List of per-epoch loss values (for plotting/inspection).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.train()

    inputs = inputs.to(device)
    labels = labels.to(device).float()
    n_targets = len(model.TARGET_NAMES)
    if labels.dim() != 2 or labels.shape[1] != n_targets:
        raise ValueError(
            f"Expected labels shape (N, {n_targets}) (one column per "
            f"QKneeMultiTargetModel.TARGET_NAMES), got {tuple(labels.shape)}"
        )

    resolved_pos_weight = (pos_weight if pos_weight is not None else compute_pos_weight(labels)).to(device)

    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=resolved_pos_weight)

    history: List[float] = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(inputs)  # (N, 12) raw logits — NOT sigmoid-activated
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if epoch % log_every == 0 or epoch == n_epochs - 1:
            logger.info("epoch %3d | loss = %.4f", epoch, loss.item())

    return history


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

CHECKPOINT_REQUIRED_KEYS = ("vqc_state_dict", "model_state_dict", "n_qubits", "n_layers")

# Canonical location for the "best" checkpoint `scripts/train.py` writes
# (see its `checkpoint_dir / "best_checkpoint.pt"` convention) under the
# name downstream inference entry points (PipelineRunner, demos, tests)
# look for first. Deliberately *not* required to exist: a fresh checkout
# with no trained checkpoint yet must still run end-to-end, backed by
# deterministic pretrained ResNet18 weights + randomly initialized quantum
# VQC parameters (see `load_best_checkpoint_or_init` below).
DEFAULT_BEST_CHECKPOINT_PATH = Path("qknee/artifacts/checkpoints/best_qknee_model.pt")


def save_checkpoint(
    model: QKneeModel,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """Saves a standardized checkpoint with two state-dict views of the same
    trained weights:

        - `vqc_state_dict`:   just the VQC's own parameters (unprefixed keys,
          e.g. `"readout.weight"`), for callers that only need the quantum
          classifier — this is what `PipelineRunner` loads.
        - `model_state_dict`: the full joint `QKneeModel` (including frozen
          ResNet/PCA buffers), for a fully self-contained reload via
          `load_checkpoint`/`QKneeModel.load_state_dict`.

    Both are always present so a checkpoint produced by this function is
    unambiguous to load regardless of which consumer reads it.
    """
    path = Path(path)
    checkpoint = {
        "vqc_state_dict": model.vqc.state_dict(),
        "model_state_dict": model.state_dict(),
        "n_qubits": model.vqc.n_qubits,
        "n_layers": model.vqc.n_layers,
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s (keys=%s)", path, sorted(checkpoint.keys()))
    return path


def _validate_checkpoint_schema(checkpoint: Dict, path: Union[str, Path]) -> None:
    """Raises a clear `KeyError`-derived error naming every missing key,
    instead of letting a malformed/legacy checkpoint fail deep inside
    `load_state_dict()` with an opaque RuntimeError."""
    missing = [key for key in CHECKPOINT_REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint at {path} is missing required key(s) {missing}. "
            f"Expected all of {CHECKPOINT_REQUIRED_KEYS} - this checkpoint may "
            "have been saved by an older/incompatible version of save_checkpoint()."
        )


def load_checkpoint(
    model: QKneeModel,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> Dict:
    """Loads full model (and optionally optimizer) state from a checkpoint
    saved by `save_checkpoint`. Mutates `model` (and `optimizer`, if given)
    in place, and restores standard PyTorch eval-mode semantics by leaving
    the module's training flag untouched (callers should call `model.eval()`
    or `model.train()` explicitly afterward, per normal PyTorch convention).

    Returns the raw checkpoint dict (useful for reading back `epoch`/`extra`).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location or "cpu")
    _validate_checkpoint_schema(checkpoint, path)

    if checkpoint["n_qubits"] != model.vqc.n_qubits:
        raise ValueError(
            f"Checkpoint was saved with n_qubits={checkpoint['n_qubits']}, "
            f"but model has n_qubits={model.vqc.n_qubits}"
        )
    if checkpoint["n_layers"] != model.vqc.n_layers:
        raise ValueError(
            f"Checkpoint was saved with n_layers={checkpoint['n_layers']}, "
            f"but model has n_layers={model.vqc.n_layers}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Loaded checkpoint from %s (epoch=%s)", path, checkpoint.get("epoch"))
    return checkpoint


def load_best_checkpoint_or_init(
    model: QKneeModel,
    path: Union[str, Path] = DEFAULT_BEST_CHECKPOINT_PATH,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> QKneeModel:
    """Loads `model`'s trained weights from `path` if it exists; otherwise
    leaves `model` as-is (deterministic pretrained ResNet18 backbone +
    randomly initialized VQC quantum parameters, exactly what `QKneeModel.
    __init__` already builds) and logs a warning instead of letting a
    missing checkpoint raise `FileNotFoundError`.

    This is the safe entry point inference/demo callers (e.g.
    `PipelineRunner`) should use instead of `load_checkpoint` directly when
    a trained checkpoint is optional (a fresh checkout with no training run
    yet should still produce a runnable, if untrained, model).
    """
    path = Path(path)
    if not path.exists():
        logger.warning(
            "[WARN] Checkpoint not found. Initialized deterministic hybrid weights for "
            "demo/eval mode. (looked for %s — proceeding with deterministic pretrained "
            "ResNet18 backbone weights and randomly initialized quantum VQC parameters.)",
            path,
        )
        return model

    load_checkpoint(model, path, optimizer=optimizer, map_location=map_location)
    return model


# --------------------------------------------------------------------------- #
# Multi-target checkpointing
# --------------------------------------------------------------------------- #

MULTITARGET_CHECKPOINT_REQUIRED_KEYS = ("head_state_dict", "model_state_dict", "n_qubits", "n_layers", "head_type", "target_names")


def save_multitarget_checkpoint(
    model: QKneeMultiTargetModel,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> Path:
    """Saves a `QKneeMultiTargetModel` checkpoint — mirrors `save_checkpoint`'s
    two-state-dict-views structure, plus `head_type`/`target_names` so a
    reload can verify it's reconstructing the *same* head architecture and
    condition ordering the checkpoint was trained with:

        - `head_state_dict`:  just the multi-target head's own parameters.
        - `model_state_dict`: the full joint model (incl. frozen ResNet/PCA
          buffers), for a fully self-contained reload.
    """
    path = Path(path)
    checkpoint = {
        "head_state_dict": model.head.state_dict(),
        "model_state_dict": model.state_dict(),
        "n_qubits": getattr(model.head, "n_qubits", None),
        "n_layers": getattr(model.head, "n_layers", None),
        "head_type": model.head_type,
        "target_names": list(model.TARGET_NAMES),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    logger.info("Saved multi-target checkpoint to %s (head_type=%s, keys=%s)", path, model.head_type, sorted(checkpoint.keys()))
    return path


def _validate_multitarget_checkpoint_schema(checkpoint: Dict, path: Union[str, Path]) -> None:
    missing = [key for key in MULTITARGET_CHECKPOINT_REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Multi-target checkpoint at {path} is missing required key(s) {missing}. "
            f"Expected all of {MULTITARGET_CHECKPOINT_REQUIRED_KEYS} - this checkpoint may "
            "have been saved by an older/incompatible version of save_multitarget_checkpoint(), "
            "or is a single-target QKneeModel checkpoint (see load_checkpoint instead)."
        )


def load_multitarget_checkpoint(
    model: QKneeMultiTargetModel,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> Dict:
    """Loads full `QKneeMultiTargetModel` (and optionally optimizer) state
    from a checkpoint saved by `save_multitarget_checkpoint`. Mutates
    `model` (and `optimizer`, if given) in place.

    Raises:
        FileNotFoundError: if `path` doesn't exist.
        KeyError: if the checkpoint is missing required keys (e.g. a
            single-target `QKneeModel` checkpoint was passed here).
        ValueError: if the checkpoint's `head_type`/`n_qubits`/`n_layers`/
            `target_names` don't match `model`'s.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=map_location or "cpu")
    _validate_multitarget_checkpoint_schema(checkpoint, path)

    if checkpoint["head_type"] != model.head_type:
        raise ValueError(
            f"Checkpoint was saved with head_type={checkpoint['head_type']!r}, "
            f"but model has head_type={model.head_type!r}"
        )
    if list(checkpoint["target_names"]) != list(model.TARGET_NAMES):
        raise ValueError(
            f"Checkpoint's target_names {checkpoint['target_names']} don't match "
            f"model.TARGET_NAMES {list(model.TARGET_NAMES)}"
        )
    model_n_qubits = getattr(model.head, "n_qubits", None)
    model_n_layers = getattr(model.head, "n_layers", None)
    if checkpoint["n_qubits"] != model_n_qubits:
        raise ValueError(f"Checkpoint was saved with n_qubits={checkpoint['n_qubits']}, but model has n_qubits={model_n_qubits}")
    if checkpoint["n_layers"] != model_n_layers:
        raise ValueError(f"Checkpoint was saved with n_layers={checkpoint['n_layers']}, but model has n_layers={model_n_layers}")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Loaded multi-target checkpoint from %s (epoch=%s, head_type=%s)", path, checkpoint.get("epoch"), checkpoint["head_type"])
    return checkpoint


# --------------------------------------------------------------------------- #
# Full forward-pass validation
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()
    torch.manual_seed(0)
    np.random.seed(0)

    # --- Fit a PCA reducer on dummy 512-D features (stand-in for a real
    #     corpus of ResNet18 embeddings fit offline) ---
    logger.info("Fitting QuantumDimReducer on dummy 512-D features...")
    dummy_512d_corpus = np.random.randn(300, 512).astype(np.float32)
    reducer = QuantumDimReducer().fit(dummy_512d_corpus)

    # --- Build the unified model ---
    model = QKneeModel(pca_reducer=reducer, n_qubits=4, n_layers=3)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.trainable_parameters())
    logger.info("Total parameters:     %d", total_params)
    logger.info("Trainable parameters: %d (VQC weights + readout only)", trainable_params)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.debug("  trainable: %s %s", name, tuple(param.shape))

    # --- Full forward-pass validation ---
    logger.info("Validating forward pass...")
    batch_size = 5
    dummy_images = torch.rand(batch_size, 3, 224, 224)
    output = model(dummy_images)

    assert output.shape == (batch_size, 1), f"Unexpected output shape {output.shape}"
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0), "Output must be in [0, 1]"
    logger.info("  Single-slice input %s -> output %s", tuple(dummy_images.shape), tuple(output.shape))
    logger.info("  Sample risk probabilities: %s", output.detach().flatten().tolist())

    dummy_volume = torch.rand(2, 6, 3, 224, 224)  # 2 volumes, 6 slices each
    volume_output = model(dummy_volume)
    assert volume_output.shape == (2, 1)
    logger.info("  Multi-slice volume input %s -> output %s", tuple(dummy_volume.shape), tuple(volume_output.shape))

    # --- Training loop utility ---
    logger.info("Running training-loop utility on synthetic labels...")
    dummy_labels = torch.randint(0, 2, (batch_size,))
    history = train_qknee_model(model, dummy_images, dummy_labels, n_epochs=25, lr=0.05, log_every=5)
    assert history[-1] < history[0], "Expected loss to decrease during training"
    logger.info("  Loss: %.4f -> %.4f", history[0], history[-1])

    # --- Checkpoint save/load round-trip ---
    logger.info("Validating checkpoint save/load...")
    checkpoint_path = Path("qknee_model.pt")
    model.eval()
    with torch.no_grad():
        pre_save_output = model(dummy_images).clone()

    save_checkpoint(model, checkpoint_path, epoch=25)

    reloaded_model = QKneeModel(pca_reducer=reducer, n_qubits=4, n_layers=3)
    load_checkpoint(reloaded_model, checkpoint_path)
    reloaded_model.eval()

    with torch.no_grad():
        post_load_output = reloaded_model(dummy_images)

    torch.testing.assert_close(pre_save_output, post_load_output, rtol=1e-5, atol=1e-6)
    logger.info("  Reloaded model produces identical output to the saved model.")

    checkpoint_path.unlink()
    logger.info("All single-target QKneeModel validations passed.")

    # --- Multi-target model validation (RSNA 12-condition head) ---
    for head_type in ("multi_observable", "ensemble"):
        logger.info("=== QKneeMultiTargetModel (head_type=%s) ===", head_type)
        torch.manual_seed(0)
        multitarget_model = QKneeMultiTargetModel(pca_reducer=reducer, n_qubits=4, n_layers=3, head_type=head_type)

        n_targets = len(multitarget_model.TARGET_NAMES)
        logits = multitarget_model(dummy_images)
        assert logits.shape == (batch_size, n_targets), f"Unexpected logits shape {tuple(logits.shape)}"
        logger.info("  forward() -> raw logits %s (NOT [0,1] -- BCEWithLogitsLoss input)", tuple(logits.shape))

        probabilities = multitarget_model.predict_proba(dummy_images)
        assert probabilities.shape == (batch_size, n_targets)
        assert torch.all(probabilities >= 0.0) and torch.all(probabilities <= 1.0), "predict_proba() must be in [0, 1]"
        logger.info("  predict_proba() -> calibrated probabilities in [0, 1]: OK")

        # --- pos_weight + BCEWithLogitsLoss training ---
        multitarget_labels = torch.randint(0, 2, (batch_size, n_targets)).float()
        pos_weight = compute_pos_weight(multitarget_labels)
        assert pos_weight.shape == (n_targets,)
        assert torch.all(pos_weight >= 0)
        logger.info("  compute_pos_weight(): %s", pos_weight.tolist())

        multitarget_history = train_qknee_multitarget_model(
            multitarget_model, dummy_images, multitarget_labels, n_epochs=20, lr=0.1, log_every=10,
        )
        assert multitarget_history[-1] < multitarget_history[0], "Expected multi-target loss to decrease during training"
        logger.info("  Multi-target loss: %.4f -> %.4f", multitarget_history[0], multitarget_history[-1])

        # --- multi-target checkpoint save/load round-trip ---
        multitarget_model.eval()
        with torch.no_grad():
            pre_save_probs = multitarget_model.predict_proba(dummy_images).clone()

        multitarget_checkpoint_path = Path(f"qknee_multitarget_model_{head_type}.pt")
        save_multitarget_checkpoint(multitarget_model, multitarget_checkpoint_path, epoch=20)

        reloaded_multitarget_model = QKneeMultiTargetModel(pca_reducer=reducer, n_qubits=4, n_layers=3, head_type=head_type)
        load_multitarget_checkpoint(reloaded_multitarget_model, multitarget_checkpoint_path)
        reloaded_multitarget_model.eval()

        with torch.no_grad():
            post_load_probs = reloaded_multitarget_model.predict_proba(dummy_images)

        torch.testing.assert_close(pre_save_probs, post_load_probs, rtol=1e-5, atol=1e-6)
        logger.info("  Reloaded multi-target model produces identical predict_proba() output.")

        multitarget_checkpoint_path.unlink()

    logger.info("All qknee_model.py validations passed.")
