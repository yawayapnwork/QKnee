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
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

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

# Target size for the MRNet-style multi-plane standardization pipeline
# (`standardize_slice` / `qknee.data.ingestion.MultiPlaneViewSelector`) —
# deliberately separate from `TARGET_SIZE`/`IMAGENET_MEAN`/`IMAGENET_STD`
# above, which stay wired to the ResNet18 backbone's 224x224 ImageNet-
# normalized input. Nothing in `build_transforms`/`MRIDataset` below reads
# this constant, so it changes independently of the ResNet pipeline.
MRNET_TARGET_SIZE: Tuple[int, int] = (128, 128)  # (height, width)


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


# --------------------------------------------------------------------------- #
# MRNet-style multi-plane standardization: resize to (3, 128, 128) + z-score
# --------------------------------------------------------------------------- #

def zscore_normalize(slice_2d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-slice z-score standardization: `(x - mean(x)) / (std(x) + eps)`.

    Per-slice (rather than dataset-wide, precomputed) statistics are
    deliberate: unlike CT's calibrated Hounsfield units, raw MRI intensities
    carry no fixed physical scale — they vary by scanner, coil, and sequence
    — so a fixed global mean/std computed on one dataset doesn't transfer
    cleanly to another. Per-slice normalization is the standard choice for
    MRNet-style multi-plane pipelines for exactly this reason.

    Args:
        slice_2d: A 2D array of any real dtype.
        eps: Added to the standard deviation to avoid a divide-by-zero on a
            constant (zero-variance) slice.

    Returns:
        `float32` array of the same shape, zero mean / unit variance
        (before `eps`).
    """
    array = np.asarray(slice_2d, dtype=np.float32)
    mean, std = float(array.mean()), float(array.std())
    return (array - mean) / (std + eps)


def standardize_slice(
    slice_2d: np.ndarray,
    target_size: Tuple[int, int] = MRNET_TARGET_SIZE,
) -> torch.Tensor:
    """Resizes one 2D grayscale MRI slice to `target_size` and z-score
    normalizes it into a `(3, H, W)` tensor — the standardization step for
    the MRNet-style multi-plane pipeline (`qknee.data.ingestion.
    MultiPlaneViewSelector`), independent of `build_transforms()`'s
    224x224/ImageNet-normalized pipeline used by the ResNet18 backbone.

    Pipeline: resize (bilinear, on the original intensity scale) ->
    per-slice z-score normalize (`zscore_normalize`) -> replicate to 3
    channels (matching a pretrained CNN's expected input arity, the same
    way `build_transforms`'s `Grayscale(num_output_channels=3)` does).

    Args:
        slice_2d: `(H, W)` array, or `(H, W, C)` with `C` in `{3, 4}`
            (collapsed to grayscale by averaging the first 3 channels).
        target_size: `(height, width)` to resize to; defaults to
            `MRNET_TARGET_SIZE` = `(128, 128)`.

    Returns:
        `(3, height, width)` `float32` tensor.

    Raises:
        ValueError: if `slice_2d` isn't 2D after channel collapsing.
    """
    array = np.asarray(slice_2d)
    if array.ndim == 3:
        if array.shape[-1] in (3, 4):
            array = array[..., :3].mean(axis=-1)
        else:
            raise ValueError(f"Expected a 2D slice or (H, W, {{3,4}}), got shape {array.shape}")
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D slice (or (H, W, C)), got shape {array.shape}")

    height, width = target_size
    # PIL's 'F' (32-bit float) mode resizes on the slice's real intensity
    # scale (no premature uint8 clipping) — important for MRI, whose raw
    # pixel values are often well outside [0, 255].
    pil_image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = pil_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    resized_array = np.asarray(resized, dtype=np.float32)

    normalized = zscore_normalize(resized_array)
    channeled = np.repeat(normalized[np.newaxis, :, :], 3, axis=0)  # (3, H, W)
    return torch.from_numpy(channeled.copy())


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


# --------------------------------------------------------------------------- #
# Mock Stanford MRNet dataset generator — for tests that need a real,
# readable on-disk dataset tree without the full (multi-GB, credentialed)
# Stanford MRNet raw dataset download.
# --------------------------------------------------------------------------- #

MRNET_PLANES: Tuple[str, str, str] = ("axial", "coronal", "sagittal")


def generate_mock_mrnet_volume(
    num_slices: int = 8, size: int = 64, seed: int = 0,
) -> np.ndarray:
    """Returns one synthetic `(num_slices, size, size)` `uint16` volume
    standing in for a single Stanford-MRNet-style per-plane `.npy` file
    (MRNet stores one `(S, 256, 256)`-ish volume per case per plane).
    Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4096, size=(num_slices, size, size), dtype=np.uint16)


def generate_mock_mrnet_dataset(
    root: Union[str, Path],
    case_ids: Sequence[str] = ("0000", "0001", "0002", "0003"),
    planes: Sequence[str] = MRNET_PLANES,
    condition: str = "acl",
    split: str = "train",
    num_slices: int = 8,
    size: int = 64,
    seed: int = 0,
) -> Path:
    """Builds a miniature, on-disk mock of the Stanford MRNet dataset's
    directory/label layout, so the test suite (and any local pipeline
    smoke-test) can exercise the multi-plane DICOM/NPY ingestion path
    end-to-end without the real dataset, which is multi-GB and requires a
    signed data-use agreement to download.

    Layout written (matches the real MRNet release's shape):

        root/
            {split}/
                axial/{case_id}.npy      (num_slices, size, size) uint16
                coronal/{case_id}.npy
                sagittal/{case_id}.npy
            {split}-{condition}.csv      "{case_id},{label}" per line, no header

    Labels are a deterministic pseudo-random 0/1 per case (seeded by
    `seed` and the case id), not clinically meaningful — this mock is for
    exercising the *data pipeline*, not for training a real model.

    Args:
        root: Directory to build the mock dataset tree under (created if
            missing).
        case_ids: Case identifiers; one `.npy` volume is written per
            `(case_id, plane)` pair.
        planes: Which of `MRNET_PLANES` to generate; defaults to all three.
        condition: Which MRNet label CSV to generate — real MRNet ships
            `train-abnormal.csv`, `train-acl.csv`, `train-meniscus.csv`.
        split: Split name (`"train"`, `"valid"`, ...), used as the
            directory/CSV-filename prefix, matching MRNet's own layout.
        num_slices: Slices per volume.
        size: Height/width per slice (kept small — this is a test fixture,
            not a realistic-resolution volume).
        seed: RNG seed; same seed reproduces the same volumes and labels.

    Returns:
        `root`, for chaining straight into a `MultiPlaneViewSelector`/
        `MRIDataset` built on top of it.
    """
    root = Path(root)
    split_dir = root / split
    label_rows: List[str] = []

    for case_offset, case_id in enumerate(case_ids):
        case_seed = seed + case_offset
        for plane in planes:
            plane_dir = split_dir / plane
            plane_dir.mkdir(parents=True, exist_ok=True)
            volume = generate_mock_mrnet_volume(num_slices=num_slices, size=size, seed=case_seed)
            np.save(plane_dir / f"{case_id}.npy", volume)

        label = int(np.random.default_rng(case_seed).integers(0, 2))
        label_rows.append(f"{case_id},{label}")

    label_csv_path = root / f"{split}-{condition}.csv"
    label_csv_path.write_text("\n".join(label_rows) + "\n", encoding="utf-8")

    logger.debug(
        "generate_mock_mrnet_dataset: wrote %d case(s) x %d plane(s) under %s, labels in %s",
        len(case_ids), len(planes), split_dir, label_csv_path,
    )
    return root


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
