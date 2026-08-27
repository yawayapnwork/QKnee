"""
Production-ready PyTorch dataset / dataloader utilities for medical MRI scans.

Supports:
    - 2D slice images on disk: .png, .jpg, .jpeg
    - 3D volumetric arrays: .npy (each file is treated as a stack of slices,
      shape (D, H, W) or (D, H, W, C); every slice is yielded as one sample)

Design:
    - `MRIDataset` does file discovery, corruption handling, and slice
      extraction from volumes. It is transform-agnostic; you pass in a
      torchvision `transforms.Compose`.
    - `build_transforms()` centralizes the preprocessing/augmentation
      pipelines for train vs. eval splits.
    - `build_dataloaders()` wires together Dataset + Transforms + DataLoader
      for train/val/test splits and returns a dict of DataLoaders.

Expected directory layout (ImageFolder-style, one subfolder per class):

    root/
        train/
            class_a/*.png|*.jpg|*.npy
            class_b/...
        val/
            class_a/...
            class_b/...
        test/
            class_a/...
            class_b/...

If your data isn't split into class subfolders (e.g. unlabeled slices),
pass `labeled=False` and the dataset will assign label -1 to every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

logger = get_logger(__name__)
_config = load_config()

IMAGE_EXTENSIONS = set(_config.data.image_extensions)
VOLUME_EXTENSIONS = set(_config.data.volume_extensions)

IMAGENET_MEAN = _config.data.imagenet_mean
IMAGENET_STD = _config.data.imagenet_std
TARGET_SIZE = _config.data.image_size


class GaussianNoise:
    """Adds zero-mean Gaussian noise to a tensor image. Applied after ToTensor,
    before Normalize, so `std` is expressed in the [0, 1] pixel-value scale."""

    def __init__(self, mean: float = 0.0, std: float = 0.02):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def build_transforms(train: bool) -> transforms.Compose:
    """Returns the preprocessing (+ augmentation, if train) pipeline.

    Pipeline order matters: resize/grayscale-to-RGB happen on the PIL image,
    then ToTensor converts to [0,1] float, then noise (train only) is added
    in tensor space, then ImageNet normalization is applied last.

    Augmentation parameters (rotation degrees, flip probability, noise std)
    are read from `config.yaml`'s `data.train_augmentation` section.
    """
    augmentation = _config.data.train_augmentation
    pipeline: List[Callable] = [
        transforms.Resize(TARGET_SIZE),
        transforms.Grayscale(num_output_channels=3),
    ]

    if train:
        pipeline += [
            transforms.RandomRotation(degrees=augmentation.random_rotation_degrees),
            transforms.RandomHorizontalFlip(p=augmentation.horizontal_flip_prob),
        ]

    pipeline.append(transforms.ToTensor())

    if train:
        pipeline.append(GaussianNoise(mean=0.0, std=augmentation.gaussian_noise_std))

    pipeline.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(pipeline)


@dataclass(frozen=True)
class Sample:
    """A reference to one loadable unit: either a whole image file, or a
    single slice index within a .npy volume."""

    path: Path
    label: int
    slice_index: Optional[int] = None  # None => 2D image file


class MRIDataset(Dataset):
    """Dataset over a directory of MRI scans (2D images and/or 3D .npy volumes).

    Corrupted or unreadable files are skipped at index-build time (logged as
    warnings) rather than crashing the whole run. A best-effort fallback also
    guards `__getitem__` in case a file becomes unreadable between indexing
    and iteration (e.g. filesystem issue, truncated write).
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        labeled: bool = True,
        extensions: Optional[Set[str]] = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.transform = transform
        self.labeled = labeled
        self.extensions = extensions or (IMAGE_EXTENSIONS | VOLUME_EXTENSIONS)

        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Sample] = []

        self._build_index()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found under {self.root} with extensions "
                f"{self.extensions}"
            )

    def _build_index(self) -> None:
        if self.labeled:
            class_dirs = sorted(p for p in self.root.iterdir() if p.is_dir())
            if not class_dirs:
                raise RuntimeError(
                    f"labeled=True but no class subdirectories found under {self.root}"
                )
            self.classes = [p.name for p in class_dirs]
            self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
            scan_dirs = [(p, self.class_to_idx[p.name]) for p in class_dirs]
        else:
            scan_dirs = [(self.root, -1)]

        for directory, label in scan_dirs:
            for file_path in sorted(directory.rglob("*")):
                if not file_path.is_file():
                    continue
                ext = file_path.suffix.lower()
                if ext not in self.extensions:
                    continue

                if ext in VOLUME_EXTENSIONS:
                    self._index_volume(file_path, label)
                else:
                    self._index_image(file_path, label)

    def _index_image(self, file_path: Path, label: int) -> None:
        try:
            with Image.open(file_path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.warning("Skipping corrupted image %s: %s", file_path, exc)
            return
        self.samples.append(Sample(path=file_path, label=label, slice_index=None))

    def _index_volume(self, file_path: Path, label: int) -> None:
        try:
            volume = np.load(file_path, mmap_mode="r")
        except (ValueError, OSError, EOFError) as exc:
            logger.warning("Skipping corrupted volume %s: %s", file_path, exc)
            return

        if volume.ndim not in (3, 4):
            logger.warning(
                "Skipping volume %s: expected 3D (D,H,W) or 4D (D,H,W,C) array, "
                "got shape %s",
                file_path,
                volume.shape,
            )
            return

        num_slices = volume.shape[0]
        for slice_index in range(num_slices):
            self.samples.append(
                Sample(path=file_path, label=label, slice_index=slice_index)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pil_image(self, sample: Sample) -> Image.Image:
        if sample.slice_index is None:
            with Image.open(sample.path) as img:
                return img.convert("L")

        volume = np.load(sample.path, mmap_mode="r")
        slice_array = np.asarray(volume[sample.slice_index])

        if slice_array.ndim == 3 and slice_array.shape[-1] in (3, 4):
            slice_array = slice_array[..., :3].mean(axis=-1)

        slice_array = slice_array.astype(np.float32)
        min_val, max_val = float(slice_array.min()), float(slice_array.max())
        if max_val > min_val:
            slice_array = (slice_array - min_val) / (max_val - min_val)
        else:
            slice_array = np.zeros_like(slice_array)
        slice_array = (slice_array * 255.0).astype(np.uint8)

        return Image.fromarray(slice_array, mode="L")

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, int]]:
        """Returns `(tensor, label)`, or `None` if the sample fails to load
        at runtime (e.g. a file corrupted/truncated after index-build time).

        Returning `None` rather than a placeholder tensor is deliberate: a
        zero-tensor stand-in would silently train the model on a fabricated
        `(blank image, real label)` pair. Pair this dataset with
        `collate_skip_invalid` (or any collate_fn that filters `None`
        entries) so invalid samples are dropped from the batch instead of
        poisoning it.
        """
        sample = self.samples[index]
        try:
            pil_image = self._load_pil_image(sample)
        except (UnidentifiedImageError, OSError, ValueError, IndexError) as exc:
            logger.error(
                "Failed to load sample %s (slice %s) at runtime: %s. "
                "Returning None so a skip-invalid collate_fn can drop it from the batch.",
                sample.path,
                sample.slice_index,
                exc,
            )
            return None

        if self.transform is not None:
            tensor = self.transform(pil_image)
        else:
            tensor = transforms.functional.to_tensor(pil_image)

        return tensor, sample.label


def collate_skip_invalid(
    batch: List[Optional[Tuple[torch.Tensor, int]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """`collate_fn` for `MRIDataset`: drops any `None` entries (samples that
    failed to load at runtime — see `MRIDataset.__getitem__`) before
    batching the rest with the standard `default_collate`.

    Returns `None` if every sample in the batch was invalid (fully filtered
    out) — callers iterating a `DataLoader` built with this collate_fn
    should skip a `None` batch rather than treat it as data, e.g.:

        for batch in loader:
            if batch is None:
                continue
            images, labels = batch
    """
    valid_samples = [item for item in batch if item is not None]
    if not valid_samples:
        logger.warning("collate_skip_invalid: entire batch of %d samples was invalid; skipping.", len(batch))
        return None
    if len(valid_samples) < len(batch):
        logger.warning(
            "collate_skip_invalid: dropped %d/%d invalid sample(s) from batch.",
            len(batch) - len(valid_samples), len(batch),
        )
    return default_collate(valid_samples)


def build_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = _config.data.batch_size,
    num_workers: int = _config.data.num_workers,
    labeled: bool = True,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """Builds Train/Val/Test DataLoaders from `data_root/{train,val,test}`.

    Any split directory that doesn't exist is silently skipped, so this also
    works for e.g. a train/test-only layout.
    """
    data_root = Path(data_root)
    split_config = {
        "train": True,
        "val": False,
        "test": False,
    }

    dataloaders: Dict[str, DataLoader] = {}

    for split, is_train in split_config.items():
        split_dir = data_root / split
        if not split_dir.exists():
            logger.info("Skipping split '%s': %s not found", split, split_dir)
            continue

        transform = build_transforms(train=is_train)
        dataset = MRIDataset(split_dir, transform=transform, labeled=labeled)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            collate_fn=collate_skip_invalid,
        )

        logger.info(
            "Loaded split '%s': %d samples, %d batches",
            split,
            len(dataset),
            len(dataloaders[split]),
        )

    return dataloaders


if __name__ == "__main__":
    import argparse

    from qknee.config.logging_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Sanity-check MRI dataloaders")
    parser.add_argument("data_root", type=str, help="Path with train/val/test subfolders")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--unlabeled", action="store_true")
    args = parser.parse_args()

    loaders = build_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        labeled=not args.unlabeled,
    )

    for split_name, loader in loaders.items():
        batch = next((b for b in loader if b is not None), None)
        if batch is None:
            logger.warning("[%s] every batch was invalid (collate_skip_invalid dropped everything)", split_name)
            continue
        images, labels = batch
        logger.info("[%s] batch shape: %s, labels: %s", split_name, tuple(images.shape), labels.tolist())
