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
import pandas as pd
import torch
import torch.nn as nn
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
# RSNA Knee dataset parser
#
# Expected on-disk layout:
#
#     train.csv / test.csv           - StudyInstanceUID + (train.csv only)
#                                       the 12 RSNA_TARGET_COLUMNS
#     <series_dir>/
#         <StudyInstanceUID>/
#             Sagittal/*.dcm             - one DICOM series per plane;
#             Coronal/*.dcm                not every study has all three
#             Axial/*.dcm
#
# `RSNAKneeDataset` parses the CSV and, for every row, resolves whichever
# of that study's plane subdirectories actually exist on disk — it never
# drops a CSV row for a missing series (see `RSNAKneeDataset`'s docstring
# for why: `scripts/generate_kaggle_submission.py` needs one output row
# per input study regardless of data availability).
# --------------------------------------------------------------------------- #

RSNA_UID_COLUMN = "StudyInstanceUID"
RSNA_PLANE_COLUMN = "Anatomical_Plane"
RSNA_SERIES_UID_COLUMN = "SeriesInstanceUID"
RSNA_PLANES: Tuple[str, str, str] = ("Sagittal", "Coronal", "Axial")

# The 12 standard RSNA Knee abnormality target columns, in the exact order
# the competition's submission schema expects
# (`scripts/generate_kaggle_submission.py` reuses this tuple verbatim for
# `submission.csv`'s column order).
RSNA_TARGET_COLUMNS: Tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


def load_rsna_labels_csv(
    csv_path: Union[str, Path],
    require_targets: bool = False,
) -> pd.DataFrame:
    """Loads an RSNA-Knee-format `train.csv`/`test.csv` into a DataFrame
    with a guaranteed, cleaned `StudyInstanceUID` column plus all 12
    `RSNA_TARGET_COLUMNS` — added as all-NaN for any that are genuinely
    absent from the file (expected for `test.csv`, which by definition
    carries no ground truth) rather than raising `KeyError` downstream.

    Robust null-handling, applied unconditionally:
        - Rows with a missing/blank `StudyInstanceUID` are dropped (logged
          as a warning) — that row can never be matched to a series
          directory or a submission row, so keeping it would only produce
          a later, harder-to-diagnose failure.
        - Each target column is coerced via `pd.to_numeric(errors="coerce")`,
          so a stray non-numeric cell becomes `NaN` instead of raising or
          silently poisoning the column's dtype.
        - Missing target columns are added as all-`NaN`, not raised on,
          unless `require_targets=True`.
        - Out-of-`[0, 1]`-range target values and duplicate
          `StudyInstanceUID`s are logged as warnings (both indicate a
          malformed source file) but do not block loading — callers that
          need to hard-fail on either should check `RSNA_TARGET_COLUMNS`/
          `RSNA_UID_COLUMN` on the returned frame themselves.

    Args:
        csv_path: Path to `train.csv` or `test.csv`.
        require_targets: If `True`, raise `ValueError` when any of the 12
            target columns is entirely absent from the file — set this
            for `train.csv`; leave `False` (default) for `test.csv`.

    Returns:
        A DataFrame with `StudyInstanceUID` (str, cleaned) plus all 12
        `RSNA_TARGET_COLUMNS` (float, `NaN` where missing/null/malformed).

    Raises:
        FileNotFoundError: if `csv_path` doesn't exist.
        ValueError: if `StudyInstanceUID` is missing from the file, or if
            `require_targets=True` and a target column is entirely absent.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RSNA labels CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if RSNA_UID_COLUMN not in df.columns:
        raise ValueError(
            f"{csv_path} is missing the required '{RSNA_UID_COLUMN}' column. "
            f"Found columns: {list(df.columns)}"
        )

    before = len(df)
    df = df[df[RSNA_UID_COLUMN].notna() & (df[RSNA_UID_COLUMN].astype(str).str.strip() != "")].copy()
    df[RSNA_UID_COLUMN] = df[RSNA_UID_COLUMN].astype(str).str.strip()
    dropped = before - len(df)
    if dropped:
        logger.warning("%s: dropped %d row(s) with missing/empty %s.", csv_path, dropped, RSNA_UID_COLUMN)

    missing_columns = [column for column in RSNA_TARGET_COLUMNS if column not in df.columns]
    if missing_columns and require_targets:
        raise ValueError(
            f"{csv_path} is missing required target column(s) {missing_columns} "
            "(require_targets=True was passed — expected for a labeled train.csv)."
        )
    if missing_columns:
        logger.debug(
            "%s has no columns for target(s) %s; filled with NaN (expected for a test.csv).",
            csv_path, missing_columns,
        )

    for column in RSNA_TARGET_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    out_of_range = {}
    for column in RSNA_TARGET_COLUMNS:
        values = df[column].dropna()
        bad = values[(values < 0) | (values > 1)]
        if not bad.empty:
            out_of_range[column] = int(len(bad))
    if out_of_range:
        logger.warning("%s has out-of-[0,1]-range target value(s): %s", csv_path, out_of_range)

    duplicate_uids = df[RSNA_UID_COLUMN][df[RSNA_UID_COLUMN].duplicated()].unique().tolist()
    if duplicate_uids:
        logger.warning(
            "%s has %d duplicate %s value(s), e.g. %s.",
            csv_path, len(duplicate_uids), RSNA_UID_COLUMN, duplicate_uids[:5],
        )

    return df.reset_index(drop=True)


def load_rsna_series_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Loads an RSNA-Knee-format `train_series.csv`/`test_series.csv` — one
    row per DICOM series, mapping each `SeriesInstanceUID` to its parent
    `StudyInstanceUID` and `Anatomical_Plane` (Sagittal/Coronal/Axial).

    This mapping is required to resolve which on-disk series directory
    belongs to which plane: the real competition's DICOM layout nests by
    `SeriesInstanceUID` — an opaque identifier bearing no relation to the
    plane name — never by a literal `Sagittal`/`Coronal`/`Axial` folder.
    See `RSNAKneeDataset`'s docstring.

    Raises:
        FileNotFoundError: if `csv_path` doesn't exist.
        ValueError: if any of `StudyInstanceUID`/`SeriesInstanceUID`/
            `Anatomical_Plane` is missing from the file.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"RSNA series CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = (RSNA_UID_COLUMN, RSNA_SERIES_UID_COLUMN, RSNA_PLANE_COLUMN)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} is missing required column(s) {missing}. Found columns: {list(df.columns)}"
        )

    df[RSNA_UID_COLUMN] = df[RSNA_UID_COLUMN].astype(str).str.strip()
    df[RSNA_SERIES_UID_COLUMN] = df[RSNA_SERIES_UID_COLUMN].astype(str).str.strip()
    return df


def discover_rsna_plane_series(
    series_dir: Union[str, Path],
    study_instance_uid: str,
    series_to_plane: Dict[str, str],
    planes: Sequence[str] = RSNA_PLANES,
) -> Dict[str, List[Path]]:
    """Resolves every DICOM series directory actually on disk under
    `<series_dir>/<study_instance_uid>/<SeriesInstanceUID>/`, grouped by
    anatomical plane via `series_to_plane` (built from
    `train_series.csv`/`test_series.csv` — see `load_rsna_series_csv`).

    Two real-data properties this must account for (neither holds for a
    literal-plane-name-folder layout):
        - The plane is series-level *metadata*, not the folder name — a
          series subdirectory is named by its opaque `SeriesInstanceUID`.
        - A study can carry more than one series for the same plane (e.g.
          distinct fluid-sensitive/fat-suppressed sequences), so this
          returns `{plane: [series_dir, ...]}` — a list per plane, not a
          single path.

    A series subdirectory not present in `series_to_plane`, or whose plane
    isn't in `planes`, is skipped. A study missing a plane entirely (or
    missing from disk altogether) simply isn't a key in the returned
    dict; this never raises for a missing/incomplete study.
    """
    from qknee.data.ingestion import DICOM_EXTENSIONS

    study_dir = Path(series_dir) / study_instance_uid
    found: Dict[str, List[Path]] = {}
    if not study_dir.is_dir():
        return found

    for series_subdir in sorted(p for p in study_dir.iterdir() if p.is_dir()):
        plane = series_to_plane.get(series_subdir.name)
        if plane is None or plane not in planes:
            continue
        if any(p.is_file() and p.suffix.lower() in DICOM_EXTENSIONS for p in series_subdir.iterdir()):
            found.setdefault(plane, []).append(series_subdir)
    return found


@dataclass(frozen=True)
class RSNAStudyRecord:
    """One row of an RSNA-Knee train/test CSV, paired with whichever of
    its per-plane DICOM series directories actually exist on disk."""

    study_instance_uid: str
    plane_series_dirs: Dict[str, List[Path]]           # {"Sagittal": [Path(...), ...], ...} — only planes found on disk
    targets: Optional[Dict[str, Optional[float]]]      # None for a test-set record; else {column: float | None}


class RSNAKneeDataset:
    """Parses an RSNA-Knee-format `train.csv`/`test.csv` (`StudyInstanceUID`
    + the 12 `RSNA_TARGET_COLUMNS` abnormality labels) and pairs every row
    with whichever DICOM series directories exist under
    `series_dir/<StudyInstanceUID>/<SeriesInstanceUID>/`, resolving each
    series's anatomical plane via the companion `train_series.csv`/
    `test_series.csv` (see `load_rsna_series_csv`) — the real competition
    layout nests by `SeriesInstanceUID`, never by a literal plane-name
    folder, and a study can carry multiple series per plane.

    Every CSV row becomes exactly one `RSNAStudyRecord`, even if none of
    its plane directories exist on disk (`plane_series_dirs` is then an
    empty dict) — a caller that needs one prediction row per input study
    (e.g. `scripts/generate_kaggle_submission.py`, which must match the
    competition's exact `StudyInstanceUID` count) can rely on
    `len(dataset) == ` the CSV's row count unconditionally; a missing
    series is the caller's problem to handle (e.g. fall back to a default
    score), not something silently dropped here.

    Args:
        csv_path: Path to `train.csv` or `test.csv`.
        series_dir: Root directory containing one subdirectory per
            `StudyInstanceUID`.
        series_csv_path: Path to the companion `train_series.csv`/
            `test_series.csv` (`StudyInstanceUID`, `SeriesInstanceUID`,
            `Anatomical_Plane` columns). Defaults to `<csv_path's stem>_series
            <csv_path's suffix>` alongside `csv_path` (i.e. `train.csv` ->
            `train_series.csv`), matching the real competition's naming.
        planes: Which `Anatomical_Plane` values to keep.
        require_targets: If `True`, raise if any of the 12 target columns
            is entirely absent from the CSV — set this for `train.csv`.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        series_dir: Union[str, Path],
        series_csv_path: Optional[Union[str, Path]] = None,
        planes: Sequence[str] = RSNA_PLANES,
        require_targets: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.series_dir = Path(series_dir)
        self.planes = tuple(planes)
        self._has_targets = require_targets or self._csv_has_any_target_columns()

        if series_csv_path is None:
            series_csv_path = self.csv_path.with_name(f"{self.csv_path.stem}_series{self.csv_path.suffix}")
        self.series_csv_path = Path(series_csv_path)
        self._series_to_plane = self._load_series_to_plane_map()

        self.frame = load_rsna_labels_csv(self.csv_path, require_targets=require_targets)
        self._records = self._build_records()

        n_missing_series = sum(1 for record in self._records if not record.plane_series_dirs)
        if n_missing_series:
            logger.warning(
                "%d/%d studies from %s have no DICOM series found under %s.",
                n_missing_series, len(self._records), self.csv_path, self.series_dir,
            )

    def _load_series_to_plane_map(self) -> Dict[str, str]:
        series_frame = load_rsna_series_csv(self.series_csv_path)
        return dict(zip(series_frame[RSNA_SERIES_UID_COLUMN], series_frame[RSNA_PLANE_COLUMN]))

    def _csv_has_any_target_columns(self) -> bool:
        try:
            header = pd.read_csv(self.csv_path, nrows=0).columns
        except Exception:
            return False
        return any(column in header for column in RSNA_TARGET_COLUMNS)

    def _build_records(self) -> List[RSNAStudyRecord]:
        records: List[RSNAStudyRecord] = []
        for _, row in self.frame.iterrows():
            uid = row[RSNA_UID_COLUMN]
            plane_dirs = discover_rsna_plane_series(self.series_dir, uid, self._series_to_plane, self.planes)

            targets: Optional[Dict[str, Optional[float]]] = None
            if self._has_targets:
                targets = {
                    column: (float(row[column]) if pd.notna(row[column]) else None)
                    for column in RSNA_TARGET_COLUMNS
                }
            records.append(RSNAStudyRecord(study_instance_uid=uid, plane_series_dirs=plane_dirs, targets=targets))
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> RSNAStudyRecord:
        return self._records[index]

    def __iter__(self):
        return iter(self._records)


# --------------------------------------------------------------------------- #
# Multi-plane series fusion
#
# A single RSNA study can carry up to three DICOM series (Sagittal,
# Coronal, Axial). Each plane is run independently through the ResNet18 ->
# PCA/quantum-dim-reduction path (`qknee.models.qknee_model.
# PCAProjectionLayer`) to its own 4-D quantum-ready embedding; not every
# study has all three planes (see `discover_rsna_plane_series`). This
# section fuses however many per-plane 4-D embeddings are present into the
# single 4-D vector fed to the 4-qubit VQC.
# --------------------------------------------------------------------------- #

PLANE_FUSION_METHODS: Tuple[str, str] = ("weighted_average", "linear_bottleneck")


class MultiPlaneEmbeddingFusion(nn.Module):
    """Fuses per-plane 4-D quantum-ready embeddings (one per available
    `RSNA_PLANES` series) into a single 4-D embedding for the 4-qubit VQC.

    Two fusion strategies:
        - `"weighted_average"`: a learned per-plane weight (softmax-
          normalized over whichever planes are actually present for a
          given study, so a missing plane doesn't dilute the others'
          weights) — output stays a convex combination of the input
          embeddings, so it never leaves their value range.
        - `"linear_bottleneck"`: concatenates up to `len(planes) *
          embedding_dim` values (missing planes zero-padded at their fixed
          slot) and projects through a single `Linear(12, 4)` layer — lets
          the fusion learn cross-plane interactions the weighted average
          can't express, at the cost of needing zero-padding to keep the
          bottleneck's input width fixed regardless of which planes are
          present.

    Args:
        planes: Canonical plane ordering; defaults to `RSNA_PLANES`
            (Sagittal, Coronal, Axial).
        embedding_dim: Dimensionality of each per-plane embedding (4, to
            match the 4-qubit VQC's input).
        method: `"weighted_average"` (default) or `"linear_bottleneck"`.
    """

    def __init__(
        self,
        planes: Sequence[str] = RSNA_PLANES,
        embedding_dim: int = 4,
        method: str = "weighted_average",
    ) -> None:
        super().__init__()
        if method not in PLANE_FUSION_METHODS:
            raise ValueError(f"method must be one of {PLANE_FUSION_METHODS}, got {method!r}")

        self.planes = tuple(planes)
        self.embedding_dim = embedding_dim
        self.method = method

        if method == "weighted_average":
            self.plane_logits = nn.Parameter(torch.zeros(len(self.planes)))
        else:  # linear_bottleneck
            self.bottleneck = nn.Linear(len(self.planes) * embedding_dim, embedding_dim)

    def forward(self, plane_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            plane_embeddings: `{plane_name: (B, embedding_dim) tensor}` —
                any non-empty subset of `self.planes`; a plane absent from
                the dict (no series found for that study, per
                `discover_rsna_plane_series`) is simply excluded from
                `"weighted_average"`, or zero-padded for
                `"linear_bottleneck"`.

        Returns:
            `(B, embedding_dim)` fused embedding.

        Raises:
            ValueError: if `plane_embeddings` is empty, or contains a key
                not in `self.planes`.
        """
        present_planes = [plane for plane in self.planes if plane in plane_embeddings]
        unknown_planes = set(plane_embeddings) - set(self.planes)
        if unknown_planes:
            raise ValueError(f"Unknown plane(s) {unknown_planes}; expected a subset of {self.planes}")
        if not present_planes:
            raise ValueError("plane_embeddings is empty — need at least one plane's embedding to fuse.")

        stacked = torch.stack([plane_embeddings[plane] for plane in present_planes], dim=1)  # (B, P, D)

        if self.method == "weighted_average":
            plane_indices = [self.planes.index(plane) for plane in present_planes]
            weights = torch.softmax(self.plane_logits[plane_indices], dim=0)  # (P,)
            return (stacked * weights.view(1, -1, 1)).sum(dim=1)  # (B, D)

        # linear_bottleneck: zero-pad any missing planes into their fixed
        # slot so the bottleneck always sees a (B, len(planes)*D) input.
        batch_size = stacked.shape[0]
        padded = torch.zeros(
            batch_size, len(self.planes), self.embedding_dim, device=stacked.device, dtype=stacked.dtype,
        )
        for local_index, plane in enumerate(present_planes):
            padded[:, self.planes.index(plane), :] = stacked[:, local_index, :]
        return self.bottleneck(padded.reshape(batch_size, -1))  # (B, D)


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
