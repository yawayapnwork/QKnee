"""
Self-supervised backbone pretraining on real, UNLABELED RSNA Knee MRI
slices — rotation prediction (Gidaris et al., 2018 "RotNet"), chosen over
a full SimCLR contrastive pipeline for CPU-only tractability at this
dataset's scale (hundreds to a few thousand real knee slices, no GPU): a
4-way classification pretext task ("was this slice rotated 0/90/180/270
degrees?") needs no large batch size, no memory bank, and no augmentation
tuning to avoid representation collapse — the three things that make
SimCLR expensive to get working well on limited compute. It still forces
the backbone to learn real anatomical structure (a network that can't
tell a rotated knee from an upright one hasn't learned much), rather than
outputting ImageNet-only, chest-X-ray/ILSVRC-shaped priors that have
never seen an actual knee MRI.

Every StudyInstanceUID's raw .dcm slices are pooled and shuffled at the
slice level (not the study level) — good for a task-agnostic pretext
objective, but callers doing anything study-aware (e.g. held-out
evaluation split by study) must partition on StudyInstanceUID themselves
before calling into this module, since RotationSliceDataset makes no such
guarantee.

Output: a plain `torch.nn.Sequential` state dict — the same
`list(resnet18(...).children())[:-1]` slice `ResNet18FeatureExtractor`
builds — saved SEPARATELY from every labeled-pipeline artifact
(`qknee/artifacts/qknee_model.pt`, `pca_scaler.pkl`, etc.), so this
pretraining step never touches or requires the labeled 58-study benchmark.
Load it back into a fresh `ResNet18FeatureExtractor` via that class's
`pretrained_backbone_path` constructor argument.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from qknee.config.logging_config import get_logger
from qknee.data.dataset import IMAGENET_MEAN, IMAGENET_STD, TARGET_SIZE
from qknee.data.ingestion import DICOM_EXTENSIONS

logger = get_logger(__name__)

N_ROTATIONS = 4  # 0, 90, 180, 270 degrees


def discover_dicom_slices(
    roots: Sequence[Union[str, Path]],
    exclude_study_uids: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Walks every root in `roots` and returns every `.dcm`/`.dicom` file
    found beneath it — the raw slice pool for self-supervised pretraining.

    Args:
        roots: One or more `train_series/`-shaped directory roots (see
            `qknee.data.dataset.RSNAKneeDataset`'s expected layout);
            duplicate StudyInstanceUIDs across roots are fine, every file
            found is still returned once (no cross-root de-duplication is
            needed since the caller passes non-overlapping OR safely-
            overlapping roots and every .dcm's path is unique regardless).
        exclude_study_uids: StudyInstanceUIDs to skip — e.g. the 58
            labeled studies, when the caller wants a pretraining pool with
            no overlap against the labeled fine-tuning/eval set (not
            required for correctness — self-supervised pretraining uses no
            labels — but keeps "pretraining data" and "eval data" cleanly
            separated for anyone auditing the split later).
    """
    exclude = set(exclude_study_uids or ())
    slices: List[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            logger.warning("discover_dicom_slices: root %s does not exist; skipping.", root)
            continue
        for study_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if study_dir.name in exclude:
                continue
            for path in study_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in DICOM_EXTENSIONS:
                    slices.append(path)
    return slices


class RotationSliceDataset(Dataset):
    """One real DICOM slice -> its 4 rotated (0/90/180/270) views, each
    resized/grayscale-replicated the same way `build_transforms` prepares
    ResNet18 input, but WITHOUT ImageNet normalization baked into the
    per-item tensor — rotation is applied in tensor space via `torch.rot90`
    (exact, no interpolation artifacts) after `Resize`+`Grayscale`+`ToTensor`,
    then normalization is applied last, same order `build_transforms` uses.

    `__getitem__` returns all 4 rotations for one slice (not one randomly
    sampled rotation), stacked as `(4, 3, H, W)` + `(4,)` labels — a
    `DataLoader`'s default collate then produces `(B, 4, 3, H, W)` /
    `(B, 4)`, flattened to `(B*4, 3, H, W)` / `(B*4,)` by the training loop.
    This "see every rotation of every slice, every epoch" design is
    deliberate for a dataset this small (hundreds to low thousands of real
    slices, not ImageNet-scale): sampling one random rotation per item
    would throw away 3/4 of the available self-supervised signal per pass.
    """

    def __init__(self, dicom_paths: Sequence[Path]):
        self.paths = list(dicom_paths)
        if not self.paths:
            raise RuntimeError("RotationSliceDataset: no DICOM slices provided.")
        # RandomResizedCrop (not a plain Resize) is deliberate here: these
        # knee MRI slices have a landscape-oriented FOV with real black
        # background concentrated on the left/right edges (measured ~86-90%
        # near-black there vs. ~46-62% top/bottom, pooled over a 60-slice
        # sample) -- a fixed, dataset-wide asymmetry that let the rotation
        # head learn "which axis is mostly black" instead of real anatomy
        # (rotation-accuracy saturated at 99%+ within 2 epochs, the
        # signature of a trivial shortcut). Cropping a randomly placed,
        # randomly scaled sub-region *before* rotation means the black
        # border's position/extent relative to the crop varies independently
        # of the applied rotation from sample to sample and epoch to epoch,
        # so that axis-asymmetry cue no longer reliably predicts the label.
        self._prepare = transforms.Compose([
            transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.4, 0.8), ratio=(0.9, 1.1)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),  # [0, 1], (3, H, W)
        ])
        self._normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_base_tensor(self, path: Path) -> Optional[torch.Tensor]:
        from qknee.data.ingestion import DataIngestion

        try:
            ingestion = DataIngestion(train=False)
            array = ingestion._load_dicom_file(path)  # calibrated (H, W) array
            pil_slices = ingestion._array_to_pil_slices(array)
            return self._prepare(pil_slices[0])
        except Exception as exc:  # noqa: BLE001 - one corrupt slice must not crash the whole epoch
            logger.warning("RotationSliceDataset: failed to load %s: %s", path, exc)
            return None

    def __getitem__(self, index: int):
        base = self._load_base_tensor(self.paths[index])
        if base is None:
            # Deterministic, cheap fallback: a mid-gray image still trains
            # the rotation head (all 4 rotations of a uniform image are
            # technically indistinguishable, so its gradient contribution
            # is ~0 either way) without crashing the batch — much simpler
            # than plumbing a `collate_skip_invalid`-style filter through
            # the 4x-expansion this dataset already does per item.
            base = torch.full((3, *TARGET_SIZE), 0.5)

        rotated = [torch.rot90(base, k=k, dims=(1, 2)) for k in range(N_ROTATIONS)]
        rotated = torch.stack([self._normalize(t) for t in rotated], dim=0)  # (4, 3, H, W)
        labels = torch.arange(N_ROTATIONS, dtype=torch.long)  # (4,)
        return rotated, labels


def _flatten_collate(batch):
    images = torch.cat([item[0] for item in batch], dim=0)  # (B*4, 3, H, W)
    labels = torch.cat([item[1] for item in batch], dim=0)  # (B*4,)
    return images, labels


@dataclass
class SSLPretrainResult:
    backbone_state_dict: dict
    n_slices: int
    n_epochs_run: int
    final_train_loss: float
    final_train_accuracy: float
    per_epoch_loss: List[float]
    per_epoch_accuracy: List[float]
    wall_clock_seconds: float


def pretrain_rotation_backbone(
    dicom_paths: Sequence[Path],
    n_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_workers: int = 0,
    seed: int = 0,
    device: str = "cpu",
    max_seconds: Optional[float] = None,
) -> SSLPretrainResult:
    """Trains a ResNet18 backbone (ImageNet-initialized, then updated end-
    to-end — not frozen, unlike every other use of `ResNet18FeatureExtractor`
    in this repo) on the 4-way rotation-prediction pretext task over
    `dicom_paths`.

    Args:
        dicom_paths: Real `.dcm` slice paths (see `discover_dicom_slices`).
        n_epochs: Training epoch ceiling.
        batch_size: Base-image batch size — actual per-step batch is
            `batch_size * 4` after rotation expansion (see
            `RotationSliceDataset`/`_flatten_collate`).
        lr: Adam learning rate. Lower than the VQC's `0.01` default
            (`config.yaml`'s `training.learning_rate`) since this updates
            every ResNet18 conv weight from an already-good ImageNet init,
            not a from-scratch small model — a large LR here risks
            catastrophically forgetting the ImageNet features this
            pretraining is meant to build on, not erase.
        num_workers: `DataLoader` worker count (0 = load in the main
            process — safest default on Windows, where multiprocessing
            workers add real overhead for a job this size).
        seed: Torch/dataloader-shuffle seed, for reproducibility.
        device: `"cpu"` (default; this repo's target deployment/dev
            environment has no CUDA) or `"cuda"`.
        max_seconds: Optional wall-clock budget — training stops after the
            epoch in progress when exceeded, same "ceiling, not a promise"
            contract as `train_quantum_vqc`'s `early_stopping_patience`.

    Returns:
        `SSLPretrainResult` — the trained backbone's own state dict
        (`nn.Sequential` matching `ResNet18FeatureExtractor`'s
        `list(resnet18(...).children())[:-1]` slice) plus training curves.
    """
    torch.manual_seed(seed)

    dataset = RotationSliceDataset(dicom_paths)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_flatten_collate, drop_last=False,
    )

    full_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    backbone = nn.Sequential(*list(full_model.children())[:-1])  # matches ResNet18FeatureExtractor
    head = nn.Linear(512, N_ROTATIONS)

    model = nn.Sequential(backbone, nn.Flatten(1), head).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    per_epoch_loss: List[float] = []
    per_epoch_accuracy: List[float] = []
    t_start = time.time()
    epochs_run = 0

    model.train()
    for epoch in range(n_epochs):
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_total = 0

        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * images.size(0)
            epoch_correct += (logits.argmax(dim=1) == labels).sum().item()
            epoch_total += images.size(0)

        epoch_loss = epoch_loss_sum / max(epoch_total, 1)
        epoch_acc = epoch_correct / max(epoch_total, 1)
        per_epoch_loss.append(epoch_loss)
        per_epoch_accuracy.append(epoch_acc)
        epochs_run = epoch + 1
        elapsed = time.time() - t_start
        logger.info(
            "[SSL rotation-pretrain] epoch %3d | loss=%.4f | rotation-acc=%.4f | elapsed=%.1fs",
            epoch, epoch_loss, epoch_acc, elapsed,
        )

        if max_seconds is not None and elapsed > max_seconds:
            logger.info(
                "[SSL rotation-pretrain] wall-clock budget (%.0fs) reached after epoch %d — stopping.",
                max_seconds, epoch,
            )
            break

    wall_clock = time.time() - t_start
    return SSLPretrainResult(
        backbone_state_dict=backbone.state_dict(),
        n_slices=len(dataset),
        n_epochs_run=epochs_run,
        final_train_loss=per_epoch_loss[-1] if per_epoch_loss else float("nan"),
        final_train_accuracy=per_epoch_accuracy[-1] if per_epoch_accuracy else float("nan"),
        per_epoch_loss=per_epoch_loss,
        per_epoch_accuracy=per_epoch_accuracy,
        wall_clock_seconds=wall_clock,
    )


def save_ssl_backbone(
    result: SSLPretrainResult,
    checkpoint_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Saves `result.backbone_state_dict` to `checkpoint_path` (a plain
    `torch.save`'d state dict, loadable via `ResNet18FeatureExtractor(...,
    pretrained_backbone_path=checkpoint_path)`) plus a JSON metadata
    sidecar (defaults to `checkpoint_path` with a `.json` suffix) recording
    the pretraining run's provenance (slice count, epochs, final
    loss/accuracy, wall-clock time) — kept alongside so anyone loading the
    checkpoint later can see exactly what it was trained on without
    re-deriving it from logs."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result.backbone_state_dict, checkpoint_path)

    metadata_path = Path(metadata_path) if metadata_path else checkpoint_path.with_suffix(".json")
    metadata = {
        "pretext_task": "rotation_prediction_4way",
        "n_slices": result.n_slices,
        "n_epochs_run": result.n_epochs_run,
        "final_train_loss": result.final_train_loss,
        "final_train_accuracy": result.final_train_accuracy,
        "per_epoch_loss": result.per_epoch_loss,
        "per_epoch_accuracy": result.per_epoch_accuracy,
        "wall_clock_seconds": result.wall_clock_seconds,
        **(extra_metadata or {}),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved SSL-pretrained backbone to %s (metadata: %s)", checkpoint_path, metadata_path)
    return checkpoint_path
