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
from typing import Iterator, List, Literal, Optional, Sequence, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from qknee.data.dataset import IMAGE_EXTENSIONS, VOLUME_EXTENSIONS, build_transforms

logger = logging.getLogger(__name__)

InputType = Union[str, Path, np.ndarray, Image.Image]
DicomSeriesInput = Sequence[Union[str, Path, object]]  # paths, or file-like uploads with .name/.read()
DicomSeriesBackend = Literal["pydicom", "sitk"]

DICOM_EXTENSIONS = {".dcm", ".dicom"}

# --------------------------------------------------------------------------- #
# Multi-plane view selection
# --------------------------------------------------------------------------- #

AnatomicalPlane = Literal["axial", "coronal", "sagittal"]

# A (D, H, W) volume's three axes correspond to the three standard
# radiological planes in this fixed order — matches
# `qknee.ui.dashboard.get_slice`'s convention, centralized here so both the
# dashboard and any offline data pipeline (this module) agree on which axis
# is which plane.
_PLANE_AXIS: dict = {"axial": 0, "coronal": 1, "sagittal": 2}


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
            return self.load_dicom_series(source)

        if isinstance(source, np.ndarray):
            return np.asarray(source)

        if isinstance(source, Image.Image):
            return np.array(source.convert("L"))

        if isinstance(source, (str, Path)):
            path = Path(source)

            if path.is_dir():
                return self.load_dicom_series(path)

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

    def load_dicom_series(
        self,
        directory_path_or_files: Union[str, Path, DicomSeriesInput],
        backend: DicomSeriesBackend = "pydicom",
    ) -> np.ndarray:
        """Reads a multi-file DICOM series into one calibrated `(D, H, W)`
        volume, sorted and stacked by `InstanceNumber` (falling back to
        `SliceLocation`, then input order) so slices stack in genuine
        anatomical order rather than filesystem/upload order.

        Args:
            directory_path_or_files: Either
                - a directory path (`str`/`Path`) containing one
                  `*.dcm`/`*.dicom` file per slice, or
                - an explicit list/tuple of DICOM file paths and/or
                  file-like uploads (e.g. Streamlit `UploadedFile`s) that
                  together make up one series.
            backend: `"pydicom"` (default) reads and sorts each file
                individually via `pydicom` — works for both a directory
                path and an in-memory list of files/uploads, and applies
                this module's own Modality LUT / MONOCHROME1 calibration
                (`_normalize_dicom_dataset`). `"sitk"` instead delegates
                the whole series (scan, GDCM-based spatial sorting, and
                Modality LUT application) to SimpleITK's
                `ImageSeriesReader` — more robust against irregular series
                (missing `InstanceNumber`, multiple series UIDs mixed in
                one folder, non-integer `ImagePositionPatient` ordering)
                at the cost of requiring a real directory on disk (GDCM
                reads files by path, not from an in-memory upload).

        Returns:
            `(D, H, W)` array, one calibrated slice per input file.

        Raises:
            IngestionError: if a directory path has no `.dcm`/`.dicom`
                files, the file list is empty, a file fails to read/decode,
                the series' slices don't share a common `(H, W)` shape, or
                (`backend="sitk"`) the input isn't a directory path.
        """
        if backend == "sitk":
            if not isinstance(directory_path_or_files, (str, Path)):
                raise IngestionError(
                    "backend='sitk' requires a directory path on disk (SimpleITK/GDCM reads "
                    "files by path); pass backend='pydicom' for an in-memory list of uploads."
                )
            return self._load_dicom_series_sitk(Path(directory_path_or_files))

        if isinstance(directory_path_or_files, (str, Path)):
            directory = Path(directory_path_or_files)
            if not directory.is_dir():
                raise IngestionError(
                    f"'{directory}' is not a directory; pass a directory of DICOM files "
                    "or a list of DICOM file paths/uploads."
                )
            file_items: DicomSeriesInput = sorted(
                p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in DICOM_EXTENSIONS
            )
            if not file_items:
                raise IngestionError(f"No .dcm/.dicom files found in directory {directory}")
        else:
            file_items = list(directory_path_or_files)
            if not file_items:
                raise IngestionError("DICOM series input is empty (no .dcm/.dicom files provided).")

        return self._read_and_stack_dicom_series(file_items)

    @staticmethod
    def _load_dicom_series_sitk(directory: Path) -> np.ndarray:
        """Reads a DICOM series directory via SimpleITK's GDCM-backed
        `ImageSeriesReader` — an alternative to `_read_and_stack_dicom_series`
        (pydicom, file-by-file) that delegates series discovery, spatial
        sorting (by `ImagePositionPatient`/`ImageOrientationPatient`, not
        just `InstanceNumber`), and Modality LUT rescaling to ITK's DICOM IO,
        which is more forgiving of irregular real-world series.

        Raises:
            IngestionError: if `directory` isn't a directory, contains no
                recognizable DICOM series, or GDCM/ITK fails to read the
                series (corrupted headers, unreadable transfer syntax, etc).
        """
        import SimpleITK as sitk

        if not directory.is_dir():
            raise IngestionError(f"'{directory}' is not a directory.")

        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory))
        except RuntimeError as exc:
            raise IngestionError(f"SimpleITK/GDCM failed to scan DICOM series in {directory}: {exc}") from exc

        if not series_ids:
            raise IngestionError(
                f"No DICOM series found in {directory} (SimpleITK/GDCM found no series UIDs — "
                "check that the directory contains valid, uncorrupted .dcm files)."
            )

        series_id = sorted(series_ids)[0]
        if len(series_ids) > 1:
            logger.warning(
                "Multiple DICOM series (%d) found in %s; using series %s. "
                "Pass a directory containing a single series to avoid this ambiguity.",
                len(series_ids), directory, series_id,
            )

        try:
            file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), series_id)
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_names)
            image = reader.Execute()
        except RuntimeError as exc:
            raise IngestionError(
                f"SimpleITK failed to read DICOM series '{series_id}' in {directory} "
                f"(corrupted header, unsupported transfer syntax, or unreadable pixel data): {exc}"
            ) from exc

        # GetArrayFromImage returns (D, H, W) for a 3D series — ITK's DICOM IO
        # has already applied RescaleSlope/RescaleIntercept and resolved
        # MONOCHROME1/MONOCHROME2, so no further calibration is needed here
        # (unlike the pydicom path, which calibrates each slice itself).
        volume = sitk.GetArrayFromImage(image)
        if volume.ndim != 3:
            raise IngestionError(
                f"Expected a 3D DICOM series volume (D, H, W) from {directory}, got shape {volume.shape}"
            )
        return np.asarray(volume)

    def _read_and_stack_dicom_series(self, file_items: DicomSeriesInput) -> np.ndarray:
        """Reads + sorts + stacks an already-resolved list of DICOM
        files/uploads into one `(D, H, W)` volume. Shared implementation
        behind `load_dicom_series`."""
        import pydicom

        file_items = list(file_items)

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


# --------------------------------------------------------------------------- #
# Multi-plane view selector pipeline
# --------------------------------------------------------------------------- #

class MultiPlaneViewSelector:
    """Wraps one `(D, H, W)` volume and selects 2D slices along any of the
    three standard radiological planes — Axial, Coronal, Sagittal — without
    re-deriving the axis convention at every call site (the dashboard's
    tri-planar viewer and any offline MRNet-style multi-plane dataset
    builder both use this instead of indexing the volume's axes by hand).

    Args:
        volume: `(D, H, W)` array — typically the output of
            `DataIngestion.load_volume_array`/`load_dicom_series`.

    Raises:
        ValueError: if `volume` isn't 3D.
    """

    def __init__(self, volume: np.ndarray) -> None:
        volume = np.asarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"MultiPlaneViewSelector expects a 3D (D, H, W) volume, got shape {volume.shape}")
        self.volume = volume

    def num_slices(self, plane: AnatomicalPlane) -> int:
        """Number of slices available along `plane` (the volume's extent
        along that plane's axis)."""
        return self.volume.shape[_PLANE_AXIS[plane]]

    def get_slice(self, plane: AnatomicalPlane, index: Optional[int] = None) -> np.ndarray:
        """Returns one 2D slice along `plane` at `index` (defaults to the
        anatomical midpoint — the most representative slice when the
        caller doesn't care which one).

        Raises:
            IndexError: if `index` is outside `[0, num_slices(plane))`.
        """
        axis = _PLANE_AXIS[plane]
        if index is None:
            index = self.num_slices(plane) // 2
        if not 0 <= index < self.num_slices(plane):
            raise IndexError(
                f"{plane} slice index {index} out of range [0, {self.num_slices(plane)}) "
                f"for volume shape {self.volume.shape}"
            )
        if plane == "axial":
            return self.volume[index, :, :]
        elif plane == "coronal":
            return self.volume[:, index, :]
        else:  # sagittal
            return self.volume[:, :, index]

    def iter_slices(self, plane: AnatomicalPlane) -> Iterator[np.ndarray]:
        """Yields every 2D slice along `plane`, in index order — the
        multi-plane analogue of iterating a volume's depth dimension, for
        building a per-plane MRNet-style `(S, H, W)` stack."""
        for index in range(self.num_slices(plane)):
            yield self.get_slice(plane, index)

    def get_slice_stack(self, plane: AnatomicalPlane) -> np.ndarray:
        """Returns every slice along `plane` stacked into one `(S, H, W)`
        array — the moral equivalent of re-orienting the volume so `plane`
        becomes the leading (depth) axis."""
        return np.stack(list(self.iter_slices(plane)), axis=0)

    def get_standardized_view(
        self, plane: AnatomicalPlane, index: Optional[int] = None, target_size: "Optional[tuple]" = None,
    ) -> "torch.Tensor":
        """Selects one slice along `plane` and standardizes it to a
        `(3, 128, 128)` z-score-normalized tensor via
        `qknee.data.dataset.standardize_slice` — the single call a
        multi-plane MRNet-style model's dataloader needs per view.
        """
        from qknee.data.dataset import standardize_slice

        kwargs = {} if target_size is None else {"target_size": target_size}
        return standardize_slice(self.get_slice(plane, index), **kwargs)

    def get_standardized_stack(self, plane: AnatomicalPlane, target_size: "Optional[tuple]" = None) -> "torch.Tensor":
        """Standardizes every slice along `plane` into one `(S, 3, 128, 128)`
        tensor stack, ready to batch through a per-view MRNet-style model."""
        import torch

        from qknee.data.dataset import standardize_slice

        kwargs = {} if target_size is None else {"target_size": target_size}
        return torch.stack([standardize_slice(s, **kwargs) for s in self.iter_slices(plane)], dim=0)


# --------------------------------------------------------------------------- #
# Mock DICOM series generator — for tests that need a real, readable DICOM
# series on disk without the multi-GB, credentialed Stanford MRNet dataset.
# --------------------------------------------------------------------------- #

def generate_mock_dicom_series(
    output_dir: Union[str, Path],
    num_slices: int = 10,
    rows: int = 64,
    columns: int = 64,
    seed: int = 0,
    modality: str = "MR",
) -> Path:
    """Writes a synthetic, structurally valid multi-file DICOM series to
    `output_dir` (one `.dcm` per slice) — readable by both
    `DataIngestion.load_dicom_series(..., backend="pydicom")` and
    `backend="sitk"`, and by any real DICOM viewer/toolkit, without
    requiring network access or the real (multi-GB, credentialed) Stanford
    MRNet raw dataset. Deterministic given `seed`, so tests built on top of
    it are reproducible.

    Each slice carries a real `SOPInstanceUID`/`SeriesInstanceUID`,
    incrementing `InstanceNumber` and `SliceLocation` (so both sort keys
    `_read_and_stack_dicom_series` looks for are present and agree), and a
    16-bit `MONOCHROME2` pixel array with an identity Modality LUT
    (`RescaleSlope=1`, `RescaleIntercept=0`), so a round-trip through
    `DataIngestion` reproduces the generated pixel values exactly.

    Args:
        output_dir: Directory to write `slice_0000.dcm`, `slice_0001.dcm`,
            ... into (created if missing; must not already contain
            `.dcm`/`.dicom` files, to avoid silently mixing series).
        num_slices: Number of slices (files) to generate.
        rows: Slice height in pixels.
        columns: Slice width in pixels.
        seed: RNG seed for the synthetic pixel data — same seed, same series.
        modality: DICOM `Modality` tag (default `"MR"`, matching this
            project's knee-MRI use case).

    Returns:
        `output_dir`, for chaining straight into
        `DataIngestion().load_dicom_series(output_dir)`.

    Raises:
        ValueError: if `output_dir` already contains `.dcm`/`.dicom` files.
    """
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in output_dir.iterdir() if p.is_file() and p.suffix.lower() in DICOM_EXTENSIONS]
    if existing:
        raise ValueError(
            f"{output_dir} already contains {len(existing)} .dcm/.dicom file(s); "
            "generate_mock_dicom_series refuses to mix a new series into an existing one."
        )

    rng = np.random.default_rng(seed)
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()

    for slice_index in range(num_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        dataset = pydicom.dataset.Dataset()
        dataset.file_meta = file_meta

        dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
        dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        dataset.SeriesInstanceUID = series_instance_uid
        dataset.StudyInstanceUID = study_instance_uid
        dataset.Modality = modality
        dataset.PatientID = "MOCK-PATIENT"
        dataset.PatientName = "Mock^Patient"

        dataset.InstanceNumber = slice_index + 1
        dataset.SliceLocation = float(slice_index)
        dataset.ImagePositionPatient = [0.0, 0.0, float(slice_index)]
        dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        dataset.PixelSpacing = [1.0, 1.0]
        dataset.SliceThickness = 1.0

        dataset.Rows = rows
        dataset.Columns = columns
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleSlope = 1.0
        dataset.RescaleIntercept = 0.0

        pixel_array = rng.integers(0, 4096, size=(rows, columns), dtype=np.uint16)
        dataset.PixelData = pixel_array.tobytes()

        dataset.save_as(output_dir / f"slice_{slice_index:04d}.dcm", enforce_file_format=True)

    logger.debug("generate_mock_dicom_series: wrote %d slices to %s", num_slices, output_dir)
    return output_dir
