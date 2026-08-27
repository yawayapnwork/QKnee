"""
Stage 1 of the Q-Knee pipeline: raw MRI slice/volume -> normalized tensor
batch ready for the ResNet18 backbone.

Accepts, interchangeably:
    - a path (str/Path) to a PNG/JPEG slice
    - a path (str/Path) to a .npy volume (multi-slice; averaged into one
      embedding after ResNet feature extraction)
    - a path (str/Path) to a single DICOM (.dcm/.dicom) file — a single
      2D slice, or a multi-frame 3D volume
    - a path (str/Path) to a directory containing a DICOM series (one
      *.dcm/*.dicom file per slice) — sorted by InstanceNumber/
      SliceLocation and stacked into one 3D (D, H, W) volume
    - a list of DICOM file paths or file-like uploads (one series, for
      callers — e.g. Streamlit's multi-file uploader — that hold the
      series in memory rather than as files in a directory on disk)
    - a path (str/Path) to a .nii/.nii.gz NIfTI volume (requires the
      optional `nibabel` dependency)
    - an in-memory np.ndarray (2D single slice, or 3D/4D volume)
    - a PIL.Image.Image

Every DICOM read (single file, series, or otherwise) applies the same
Modality LUT (`RescaleSlope`/`RescaleIntercept`, or a full
`ModalityLUTSequence`) and MONOCHROME1-inversion calibration that
`qknee.api.server` applies for single-slice uploads, so intensity
semantics stay consistent regardless of entry point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from qknee.data.dataset import IMAGE_EXTENSIONS, VOLUME_EXTENSIONS, build_transforms

logger = logging.getLogger(__name__)

InputType = Union[str, Path, np.ndarray, Image.Image]
DicomSeriesInput = Sequence[Union[str, Path, object]]  # paths, or file-like uploads with .name/.read()

DICOM_EXTENSIONS = {".dcm", ".dicom"}


class IngestionError(RuntimeError):
    """Raised when raw input cannot be normalized into a model-ready tensor."""


class DataIngestion:
    """Normalizes heterogeneous MRI input (path, array, PIL image, DICOM
    series, NIfTI volume) into a `(1, S, 3, 224, 224)` tensor batch for
    the ResNet18 stage.

    Args:
        train: Whether to apply the training-time augmentation pipeline
            (see `qknee.data.dataset.build_transforms`). Inference call sites
            should leave this False.
    """

    def __init__(self, train: bool = False) -> None:
        self.transform = build_transforms(train=train)

    def load_slices_as_pil(self, source: InputType) -> List[Image.Image]:
        """Normalizes any accepted input type into a list of grayscale PIL
        images (one per slice; length 1 for a single 2D image). See
        `load_volume_array` for the full list of accepted formats."""
        if isinstance(source, Image.Image):
            return [source.convert("L")]

        array = self.load_volume_array(source)
        return self._array_to_pil_slices(array)

    def load_volume_array(self, source: Union[InputType, DicomSeriesInput]) -> np.ndarray:
        """Loads `source` into a raw numpy array: `(H, W)` for a single
        slice, or `(D, H, W)` / `(D, H, W, C)` for a volume — without PIL
        slice-list conversion or transform normalization. This is the
        shared loading step behind `load_slices_as_pil`, and is also used
        directly by callers that need real volume indexing along all
        three axes (e.g. the dashboard's tri-planar Axial/Coronal/
        Sagittal slicing), not a flat list of per-depth PIL slices.

        Accepts:
            - an in-memory np.ndarray or PIL.Image.Image (passed through /
              converted directly)
            - a path to a .npy volume, or a PNG/JPEG image file
            - a path to a single .dcm/.dicom file (2D slice, or a
              multi-frame 3D volume)
            - a path to a directory containing a DICOM series
              (*.dcm/*.dicom files, sorted into one 3D volume)
            - a list/tuple of DICOM file paths or file-like uploads (one
              series, stacked the same way as the directory case)
            - a path to a .nii/.nii.gz NIfTI volume (requires `nibabel`)

        Raises:
            IngestionError: on a missing/corrupt/unsupported input, with a
                message naming the offending file and format.
        """
        if isinstance(source, (list, tuple)):
            return self._load_dicom_series(source)

        if isinstance(source, np.ndarray):
            return np.asarray(source)

        if isinstance(source, Image.Image):
            return np.array(source.convert("L"))

        if isinstance(source, (str, Path)):
            path = Path(source)

            if path.is_dir():
                dicom_files = sorted(
                    p for p in path.iterdir() if p.is_file() and p.suffix.lower() in DICOM_EXTENSIONS
                )
                if not dicom_files:
                    raise IngestionError(f"No .dcm/.dicom files found in directory {path}")
                return self._load_dicom_series(dicom_files)

            if not path.exists():
                raise IngestionError(f"Input file does not exist: {path}")

            ext = path.suffix.lower()

            if ext in DICOM_EXTENSIONS:
                return self._load_dicom_file(path)

            if ext == ".nii" or path.name.lower().endswith(".nii.gz"):
                return self._load_nifti(path)

            if ext in VOLUME_EXTENSIONS:
                try:
                    return np.load(path)
                except (ValueError, OSError, EOFError) as exc:
                    raise IngestionError(f"Corrupted/unreadable .npy volume {path}: {exc}") from exc

            if ext in IMAGE_EXTENSIONS:
                try:
                    with Image.open(path) as img:
                        img.load()
                        return np.array(img.convert("L"))
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise IngestionError(f"Corrupted/unreadable image {path}: {exc}") from exc

            raise IngestionError(
                f"Unsupported file extension '{ext}' for {path}. Expected one of "
                f"{IMAGE_EXTENSIONS | VOLUME_EXTENSIONS | DICOM_EXTENSIONS | {'.nii', '.nii.gz'}}, "
                "or a directory containing a DICOM series."
            )

        if hasattr(source, "read"):
            # A single file-like upload (e.g. one Streamlit `UploadedFile`)
            # rather than a filesystem path — dispatch by its `.name` the
            # same way the on-disk-path branch above dispatches by suffix.
            name = getattr(source, "name", "<uploaded file>")
            suffix = Path(name).suffix.lower()
            if hasattr(source, "seek"):
                source.seek(0)

            if suffix in DICOM_EXTENSIONS:
                return self._load_dicom_file(source, display_name=name)

            if suffix == ".nii" or name.lower().endswith(".nii.gz"):
                return self._load_nifti(source, display_name=name)

            if suffix in VOLUME_EXTENSIONS:
                try:
                    return np.load(source)
                except (ValueError, OSError, EOFError) as exc:
                    raise IngestionError(f"Corrupted/unreadable .npy volume '{name}': {exc}") from exc

            if suffix in IMAGE_EXTENSIONS:
                try:
                    img = Image.open(source)
                    img.load()
                    return np.array(img.convert("L"))
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    raise IngestionError(f"Corrupted/unreadable image '{name}': {exc}") from exc

            raise IngestionError(
                f"Unsupported file extension '{suffix}' for uploaded file '{name}'. Expected one of "
                f"{IMAGE_EXTENSIONS | VOLUME_EXTENSIONS | DICOM_EXTENSIONS | {'.nii', '.nii.gz'}}."
            )

        raise IngestionError(
            f"Unsupported input type {type(source)}. Expected str, Path, np.ndarray, "
            "PIL.Image.Image, a file-like upload, or a list of DICOM series files."
        )

    # ------------------------------------------------------------------ #
    # DICOM (single file, directory series, or in-memory file list)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_dicom_dataset(dataset) -> np.ndarray:
        """Applies the Modality LUT (`RescaleSlope`/`RescaleIntercept`, or
        a full `ModalityLUTSequence` if present) and inverts `MONOCHROME1`
        datasets to the `MONOCHROME2` convention — the same calibration
        `qknee.api.server.QKneeBackend._load_dicom_slice` applies for
        single-slice uploads, factored out here so series/volume DICOM
        reads decode with identical intensity semantics."""
        from pydicom.pixels import apply_modality_lut

        array = apply_modality_lut(dataset.pixel_array, dataset)
        if getattr(dataset, "PhotometricInterpretation", None) == "MONOCHROME1":
            array = np.asarray(array)
            array = array.max() - array
        return np.asarray(array)

    def _load_dicom_file(self, source: Union[Path, object], display_name: Optional[str] = None) -> np.ndarray:
        """Reads one .dcm/.dicom file (a `Path`, or a file-like upload) into
        a calibrated array — 2D for a single-frame slice, or 3D for a
        multi-frame (enhanced) DICOM."""
        import pydicom

        display_name = display_name or str(source)
        try:
            dataset = pydicom.dcmread(source if hasattr(source, "read") else str(source), force=True)
        except Exception as exc:
            raise IngestionError(f"Failed to read DICOM file '{display_name}': {exc}") from exc

        try:
            return self._normalize_dicom_dataset(dataset)
        except Exception as exc:
            raise IngestionError(f"Failed to decode pixel data in '{display_name}': {exc}") from exc

    def _load_dicom_series(self, file_items: DicomSeriesInput) -> np.ndarray:
        """Reads a list of single-frame DICOM files/uploads (one series)
        into one calibrated `(D, H, W)` volume, ordered by `InstanceNumber`
        (falling back to `SliceLocation`, then upload order) so slices
        stack in genuine anatomical order rather than upload/filesystem
        order."""
        import pydicom

        file_items = list(file_items)
        if not file_items:
            raise IngestionError("DICOM series input is empty (no .dcm/.dicom files provided).")

        entries: List[tuple] = []
        for index, item in enumerate(file_items):
            name = getattr(item, "name", str(item))
            try:
                if hasattr(item, "read"):
                    if hasattr(item, "seek"):
                        item.seek(0)
                    dataset = pydicom.dcmread(item, force=True)
                else:
                    dataset = pydicom.dcmread(str(item), force=True)
            except Exception as exc:
                raise IngestionError(f"Failed to read DICOM series file '{name}': {exc}") from exc

            sort_key = getattr(dataset, "InstanceNumber", None)
            if sort_key is None:
                sort_key = getattr(dataset, "SliceLocation", None)
            if sort_key is None:
                sort_key = index

            try:
                array = self._normalize_dicom_dataset(dataset)
            except Exception as exc:
                raise IngestionError(f"Failed to decode pixel data in '{name}': {exc}") from exc

            entries.append((float(sort_key), array))

        entries.sort(key=lambda pair: pair[0])
        arrays = [array for _, array in entries]

        shapes = {array.shape for array in arrays}
        if len(shapes) > 1:
            raise IngestionError(
                f"DICOM series slices have mismatched shapes ({shapes}); "
                "expected every slice in a series to share the same (H, W)."
            )

        return np.stack(arrays, axis=0)

    # ------------------------------------------------------------------ #
    # NIfTI
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_nifti(source: Union[Path, object], display_name: Optional[str] = None) -> np.ndarray:
        """Reads a .nii/.nii.gz volume (a `Path`, or a file-like upload)
        into a `(D, H, W)` array, via the optional `nibabel` dependency.
        NIfTI's on-disk axis convention is `(X, Y, Z)`; the last axis is
        moved first here so this method returns the same depth-first
        `(D, H, W)` convention every other volume format in this module
        uses.

        A file-like `source` (no real path on disk — e.g. a Streamlit
        upload) is spooled to a temp file first: `nibabel`'s gzip-aware
        NIfTI reader needs to seek a real file, not an in-memory stream.
        """
        try:
            import nibabel as nib
        except ImportError as exc:
            raise IngestionError(
                "Reading .nii/.nii.gz volumes requires the optional 'nibabel' "
                "dependency (pip install nibabel)."
            ) from exc

        display_name = display_name or str(source)
        try:
            if hasattr(source, "read"):
                import os
                import tempfile

                suffix = ".nii.gz" if display_name.lower().endswith(".nii.gz") else ".nii"
                if hasattr(source, "seek"):
                    source.seek(0)
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                    tmp_file.write(source.read())
                    tmp_path = tmp_file.name
                try:
                    data = np.asarray(nib.load(tmp_path).dataobj)
                finally:
                    os.unlink(tmp_path)
            else:
                data = np.asarray(nib.load(str(source)).dataobj)
        except Exception as exc:
            raise IngestionError(f"Failed to read NIfTI volume '{display_name}': {exc}") from exc

        if data.ndim != 3:
            raise IngestionError(f"Expected a 3D NIfTI volume (X, Y, Z), got shape {data.shape}")

        return np.transpose(data, (2, 0, 1))  # (X, Y, Z) -> (D, H, W)

    # ------------------------------------------------------------------ #
    # Array -> PIL slice list, and final tensor preprocessing
    # ------------------------------------------------------------------ #

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

    def preprocess(self, source: InputType) -> "torch.Tensor":
        """Returns a `(1, S, 3, 224, 224)` tensor ready for the ResNet18 backbone."""
        import torch

        pil_slices = self.load_slices_as_pil(source)
        if not pil_slices:
            raise IngestionError("Input produced zero usable slices.")

        try:
            tensors = [self.transform(img) for img in pil_slices]
        except Exception as exc:
            raise IngestionError(f"Preprocessing/transform stage failed: {exc}") from exc

        stacked = torch.stack(tensors, dim=0)  # (S, 3, 224, 224)
        return stacked.unsqueeze(0)  # (1, S, 3, 224, 224)
