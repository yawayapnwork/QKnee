"""
Stage 1 of the Q-Knee pipeline: raw MRI slice/volume -> normalized tensor
batch ready for the ResNet18 backbone.

Accepts, interchangeably:
    - a path (str/Path) to a PNG/JPEG slice
    - a path (str/Path) to a .npy volume (multi-slice; averaged into one
      embedding after ResNet feature extraction)
    - an in-memory np.ndarray (2D single slice, or 3D/4D volume)
    - a PIL.Image.Image
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from qknee.data.dataset import IMAGE_EXTENSIONS, VOLUME_EXTENSIONS, build_transforms

logger = logging.getLogger(__name__)

InputType = Union[str, Path, np.ndarray, Image.Image]


class IngestionError(RuntimeError):
    """Raised when raw input cannot be normalized into a model-ready tensor."""


class DataIngestion:
    """Normalizes heterogeneous MRI input (path, array, PIL image) into a
    `(1, S, 3, 224, 224)` tensor batch for the ResNet18 stage.

    Args:
        train: Whether to apply the training-time augmentation pipeline
            (see `qknee.data.dataset.build_transforms`). Inference call sites
            should leave this False.
    """

    def __init__(self, train: bool = False) -> None:
        self.transform = build_transforms(train=train)

    def load_slices_as_pil(self, source: InputType) -> List[Image.Image]:
        """Normalizes any accepted input type into a list of grayscale PIL
        images (one per slice; length 1 for a single 2D image)."""
        if isinstance(source, Image.Image):
            return [source.convert("L")]

        if isinstance(source, np.ndarray):
            return self._array_to_pil_slices(source)

        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise IngestionError(f"Input file does not exist: {path}")

            ext = path.suffix.lower()
            if ext in VOLUME_EXTENSIONS:
                try:
                    volume = np.load(path)
                except (ValueError, OSError, EOFError) as exc:
                    raise IngestionError(f"Corrupted/unreadable .npy volume {path}: {exc}") from exc
                return self._array_to_pil_slices(volume)

            if ext in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as img:
                        img.load()
                        return [img.convert("L")]
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise IngestionError(f"Corrupted/unreadable image {path}: {exc}") from exc

            raise IngestionError(
                f"Unsupported file extension '{ext}' for {path}. "
                f"Expected one of {IMAGE_EXTENSIONS | VOLUME_EXTENSIONS}"
            )

        raise IngestionError(
            f"Unsupported input type {type(source)}. Expected str, Path, "
            "np.ndarray, or PIL.Image.Image."
        )

    @staticmethod
    def _array_to_pil_slices(array: np.ndarray) -> List[Image.Image]:
        array = np.asarray(array)

        if array.ndim == 2:
            slices = [array]
        elif array.ndim in (3, 4):
            slices = [np.asarray(array[i]) for i in range(array.shape[0])]
        else:
            raise IngestionError(f"Expected a 2D slice or 3D/4D volume array, got shape {array.shape}")

        pil_slices = []
        for slice_array in slices:
            if slice_array.ndim == 3 and slice_array.shape[-1] in (3, 4):
                slice_array = slice_array[..., :3].mean(axis=-1)
            slice_array = slice_array.astype(np.float32)

            min_val, max_val = float(slice_array.min()), float(slice_array.max())
            if max_val > min_val:
                slice_array = (slice_array - min_val) / (max_val - min_val)
            else:
                slice_array = np.zeros_like(slice_array)

            slice_array = (slice_array * 255.0).astype(np.uint8)
            pil_slices.append(Image.fromarray(slice_array, mode="L"))

        return pil_slices

    def preprocess(self, source: InputType) -> torch.Tensor:
        """Returns a `(1, S, 3, 224, 224)` tensor ready for the ResNet18 backbone."""
        pil_slices = self.load_slices_as_pil(source)
        if not pil_slices:
            raise IngestionError("Input produced zero usable slices.")

        try:
            tensors = [self.transform(img) for img in pil_slices]
        except Exception as exc:
            raise IngestionError(f"Preprocessing/transform stage failed: {exc}") from exc

        stacked = torch.stack(tensors, dim=0)  # (S, 3, 224, 224)
        return stacked.unsqueeze(0)  # (1, S, 3, 224, 224)
