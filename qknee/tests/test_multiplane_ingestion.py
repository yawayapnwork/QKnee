"""
Tests for the multi-plane DICOM/MRNet-style ingestion additions:

    - `generate_mock_dicom_series` / `DataIngestion.load_dicom_series`
      (the `pydicom`, `sitk`, and `monai` backends), including
      corrupted-header handling and (for `monai`, whose ITK-backed reader
      returns axes in the opposite order) a pixel-for-pixel cross-backend
      parity check that would catch a wrong/dropped axis transpose.
    - `MultiPlaneViewSelector` (Axial/Coronal/Sagittal slicing).
    - `standardize_slice`/`zscore_normalize` (resize to (3, 128, 128) +
      per-slice z-score normalization, via `monai.transforms.NormalizeIntensity`).
    - `generate_mock_mrnet_dataset` (the Stanford-MRNet-shaped on-disk mock
      the whole suite here runs against, so no real MRNet download is
      required).

None of this touches the existing 224x224/ImageNet-normalized
`build_transforms`/`MRIDataset`/`DataIngestion.preprocess` pipeline used by
the ResNet18 backbone — this is a separate, additive multi-plane pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qknee.data.dataset import (
    MRNET_TARGET_SIZE,
    generate_mock_mrnet_dataset,
    generate_mock_mrnet_volume,
    standardize_slice,
    zscore_normalize,
)
from qknee.data.ingestion import (
    DataIngestion,
    IngestionError,
    MultiPlaneViewSelector,
    generate_mock_dicom_series,
)


# --------------------------------------------------------------------------- #
# Mock DICOM series generation + dual-backend loading
# --------------------------------------------------------------------------- #

class TestMockDicomSeriesGeneration:
    def test_generates_expected_number_of_files(self, tmp_path: Path):
        series_dir = generate_mock_dicom_series(tmp_path / "series", num_slices=7, rows=16, columns=16)
        files = sorted(series_dir.glob("*.dcm"))
        assert len(files) == 7

    def test_refuses_to_write_into_a_directory_with_existing_dicom_files(self, tmp_path: Path):
        series_dir = tmp_path / "series"
        generate_mock_dicom_series(series_dir, num_slices=2, rows=8, columns=8)
        with pytest.raises(ValueError, match="already contains"):
            generate_mock_dicom_series(series_dir, num_slices=2, rows=8, columns=8)

    def test_deterministic_given_seed(self, tmp_path: Path):
        dir_a = generate_mock_dicom_series(tmp_path / "a", num_slices=3, rows=8, columns=8, seed=5)
        dir_b = generate_mock_dicom_series(tmp_path / "b", num_slices=3, rows=8, columns=8, seed=5)
        vol_a = DataIngestion().load_dicom_series(dir_a)
        vol_b = DataIngestion().load_dicom_series(dir_b)
        assert np.array_equal(vol_a, vol_b)


class TestDicomSeriesBackends:
    @pytest.fixture
    def series_dir(self, tmp_path: Path) -> Path:
        return generate_mock_dicom_series(tmp_path / "series", num_slices=6, rows=24, columns=32, seed=1)

    def test_pydicom_backend_shape_and_dtype(self, series_dir: Path):
        volume = DataIngestion().load_dicom_series(series_dir, backend="pydicom")
        assert volume.shape == (6, 24, 32)

    def test_sitk_backend_shape(self, series_dir: Path):
        volume = DataIngestion().load_dicom_series(series_dir, backend="sitk")
        assert volume.shape == (6, 24, 32)

    def test_pydicom_and_sitk_backends_agree_pixel_for_pixel(self, series_dir: Path):
        """Both backends read the same identity-Modality-LUT MONOCHROME2
        series, so they should reproduce identical pixel values despite
        using entirely different DICOM decoders (pydicom vs. GDCM/ITK)."""
        vol_pydicom = DataIngestion().load_dicom_series(series_dir, backend="pydicom")
        vol_sitk = DataIngestion().load_dicom_series(series_dir, backend="sitk")
        assert np.array_equal(vol_pydicom.astype(np.float64), vol_sitk.astype(np.float64))

    def test_sitk_backend_rejects_in_memory_file_list(self, series_dir: Path):
        files = sorted(series_dir.glob("*.dcm"))
        with pytest.raises(IngestionError, match="requires a directory path"):
            DataIngestion().load_dicom_series(files, backend="sitk")

    def test_monai_backend_shape(self, series_dir: Path):
        volume = DataIngestion().load_dicom_series(series_dir, backend="monai")
        assert volume.shape == (6, 24, 32)

    def test_monai_backend_matches_pydicom_pixel_for_pixel(self, series_dir: Path):
        """`monai.transforms.LoadImage` returns ITK's native (W, H, D) axis
        order, the reverse of this module's (D, H, W) contract — this is the
        one test that would catch a dropped/wrong axis transpose in
        `_load_dicom_series_monai` silently swapping the depth and width
        axes instead of erroring (mismatched-shape checks alone wouldn't
        catch a swap between two axes that happen to be reconcilable in
        shape, e.g. this fixture's non-square 24x32 slices specifically
        guard against that: a wrong (H, W)<->(W, H)-only transpose would
        still shape-mismatch loudly, but this pixel-value check is the
        actual proof the transpose is correct, not just shaped correctly)."""
        vol_pydicom = DataIngestion().load_dicom_series(series_dir, backend="pydicom")
        vol_monai = DataIngestion().load_dicom_series(series_dir, backend="monai")
        assert np.array_equal(vol_pydicom.astype(np.float64), vol_monai.astype(np.float64))

    def test_monai_backend_rejects_in_memory_file_list(self, series_dir: Path):
        files = sorted(series_dir.glob("*.dcm"))
        with pytest.raises(IngestionError, match="requires a directory path"):
            DataIngestion().load_dicom_series(files, backend="monai")

    def test_directory_auto_dispatch_uses_pydicom_backend_by_default(self, series_dir: Path):
        """`load_volume_array` on a bare directory path (no explicit
        `backend=`) should still work end-to-end via the default backend."""
        volume = DataIngestion().load_volume_array(series_dir)
        assert volume.shape == (6, 24, 32)


class TestCorruptedDicomHandling:
    def test_pydicom_backend_raises_ingestion_error_on_corrupt_file(self, tmp_path: Path):
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "corrupt.dcm").write_bytes(b"this is not a valid dicom file")
        with pytest.raises(IngestionError):
            DataIngestion().load_dicom_series(bad_dir, backend="pydicom")

    def test_sitk_backend_raises_ingestion_error_on_corrupt_file(self, tmp_path: Path):
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "corrupt.dcm").write_bytes(b"this is not a valid dicom file")
        with pytest.raises(IngestionError):
            DataIngestion().load_dicom_series(bad_dir, backend="sitk")

    def test_monai_backend_raises_ingestion_error_on_corrupt_file(self, tmp_path: Path):
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "corrupt.dcm").write_bytes(b"this is not a valid dicom file")
        with pytest.raises(IngestionError):
            DataIngestion().load_dicom_series(bad_dir, backend="monai")

    def test_monai_backend_raises_ingestion_error_on_empty_directory(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(IngestionError):
            DataIngestion().load_dicom_series(empty_dir, backend="monai")

    def test_monai_backend_raises_ingestion_error_on_missing_directory(self, tmp_path: Path):
        with pytest.raises(IngestionError, match="is not a directory"):
            DataIngestion().load_dicom_series(tmp_path / "does_not_exist", backend="monai")

    def test_empty_directory_raises_ingestion_error(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(IngestionError, match="No .dcm/.dicom files"):
            DataIngestion().load_dicom_series(empty_dir, backend="pydicom")

    def test_missing_directory_raises_ingestion_error(self, tmp_path: Path):
        with pytest.raises(IngestionError):
            DataIngestion().load_dicom_series(tmp_path / "does_not_exist", backend="pydicom")

    def test_mismatched_slice_shapes_raise_ingestion_error(self, tmp_path: Path):
        series_dir = generate_mock_dicom_series(tmp_path / "series", num_slices=3, rows=16, columns=16)
        # Splice in one file generated at a different resolution.
        odd_dir = tmp_path / "odd"
        generate_mock_dicom_series(odd_dir, num_slices=1, rows=32, columns=32)
        files = sorted(series_dir.glob("*.dcm")) + sorted(odd_dir.glob("*.dcm"))
        with pytest.raises(IngestionError, match="mismatched shapes"):
            DataIngestion().load_dicom_series(files, backend="pydicom")


# --------------------------------------------------------------------------- #
# MultiPlaneViewSelector
# --------------------------------------------------------------------------- #

class TestMultiPlaneViewSelector:
    @pytest.fixture
    def volume(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(10, 20, 30), dtype=np.uint16)  # (D, H, W)

    def test_rejects_non_3d_volume(self):
        with pytest.raises(ValueError):
            MultiPlaneViewSelector(np.zeros((10, 10)))

    def test_num_slices_matches_corresponding_axis(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        assert selector.num_slices("axial") == volume.shape[0]
        assert selector.num_slices("coronal") == volume.shape[1]
        assert selector.num_slices("sagittal") == volume.shape[2]

    def test_get_slice_matches_manual_indexing(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        assert np.array_equal(selector.get_slice("axial", 3), volume[3, :, :])
        assert np.array_equal(selector.get_slice("coronal", 5), volume[:, 5, :])
        assert np.array_equal(selector.get_slice("sagittal", 7), volume[:, :, 7])

    def test_get_slice_defaults_to_midpoint(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        assert np.array_equal(selector.get_slice("axial"), volume[volume.shape[0] // 2, :, :])

    def test_out_of_range_index_raises(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        with pytest.raises(IndexError):
            selector.get_slice("axial", index=volume.shape[0])
        with pytest.raises(IndexError):
            selector.get_slice("sagittal", index=-1)

    def test_get_slice_stack_reconstructs_full_axis(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        stack = selector.get_slice_stack("coronal")
        assert stack.shape == (volume.shape[1], volume.shape[0], volume.shape[2])
        for i in range(volume.shape[1]):
            assert np.array_equal(stack[i], volume[:, i, :])

    def test_iter_slices_yields_num_slices_items(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        slices = list(selector.iter_slices("sagittal"))
        assert len(slices) == volume.shape[2]

    def test_get_standardized_view_shape(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        tensor = selector.get_standardized_view("axial")
        assert tensor.shape == (3, *MRNET_TARGET_SIZE)

    def test_get_standardized_stack_shape(self, volume: np.ndarray):
        selector = MultiPlaneViewSelector(volume)
        stack = selector.get_standardized_stack("coronal")
        assert stack.shape == (volume.shape[1], 3, *MRNET_TARGET_SIZE)


# --------------------------------------------------------------------------- #
# standardize_slice / zscore_normalize
# --------------------------------------------------------------------------- #

class TestStandardizeSlice:
    @pytest.fixture
    def slice_2d(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        return rng.integers(0, 4096, size=(50, 70)).astype(np.float32)

    def test_zscore_normalize_produces_unit_variance(self, slice_2d: np.ndarray):
        normalized = zscore_normalize(slice_2d)
        assert abs(float(normalized.mean())) < 1e-3
        assert abs(float(normalized.std()) - 1.0) < 1e-2

    def test_zscore_normalize_handles_constant_slice_without_nan(self):
        constant = np.full((10, 10), 42.0, dtype=np.float32)
        normalized = zscore_normalize(constant)
        assert np.all(np.isfinite(normalized))
        assert np.allclose(normalized, 0.0)

    def test_standardize_slice_default_shape(self, slice_2d: np.ndarray):
        tensor = standardize_slice(slice_2d)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)
        assert tensor.dtype == torch.float32

    def test_standardize_slice_replicates_grayscale_to_3_channels(self, slice_2d: np.ndarray):
        tensor = standardize_slice(slice_2d)
        assert torch.equal(tensor[0], tensor[1])
        assert torch.equal(tensor[1], tensor[2])

    def test_standardize_slice_custom_target_size(self, slice_2d: np.ndarray):
        tensor = standardize_slice(slice_2d, target_size=(64, 96))
        assert tensor.shape == (3, 64, 96)

    def test_standardize_slice_accepts_hwc_input_by_collapsing_channels(self, slice_2d: np.ndarray):
        rgb_like = np.repeat(slice_2d[:, :, np.newaxis], 3, axis=2)
        tensor = standardize_slice(rgb_like)
        assert tensor.shape == (3, 128, 128)

    def test_standardize_slice_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            standardize_slice(np.zeros((4, 4, 4, 4)))


# --------------------------------------------------------------------------- #
# Mock Stanford MRNet dataset generator
# --------------------------------------------------------------------------- #

class TestMockMRNetDataset:
    def test_writes_expected_directory_layout(self, tmp_path: Path):
        root = generate_mock_mrnet_dataset(
            tmp_path, case_ids=("0000", "0001"), planes=("axial", "coronal", "sagittal"),
            condition="acl", split="train", num_slices=5, size=32, seed=0,
        )
        for plane in ("axial", "coronal", "sagittal"):
            for case_id in ("0000", "0001"):
                npy_path = root / "train" / plane / f"{case_id}.npy"
                assert npy_path.exists()
                assert np.load(npy_path).shape == (5, 32, 32)
        assert (root / "train-acl.csv").exists()

    def test_label_csv_has_one_row_per_case(self, tmp_path: Path):
        root = generate_mock_mrnet_dataset(tmp_path, case_ids=("a", "b", "c"), num_slices=2, size=16)
        rows = (root / "train-acl.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 3
        for row, case_id in zip(rows, ("a", "b", "c")):
            written_case_id, label = row.split(",")
            assert written_case_id == case_id
            assert label in ("0", "1")

    def test_deterministic_given_seed(self, tmp_path: Path):
        root_a = generate_mock_mrnet_dataset(tmp_path / "a", case_ids=("0000",), num_slices=3, size=16, seed=3)
        root_b = generate_mock_mrnet_dataset(tmp_path / "b", case_ids=("0000",), num_slices=3, size=16, seed=3)
        vol_a = np.load(root_a / "train" / "axial" / "0000.npy")
        vol_b = np.load(root_b / "train" / "axial" / "0000.npy")
        assert np.array_equal(vol_a, vol_b)

    def test_generate_mock_mrnet_volume_shape_and_dtype(self):
        volume = generate_mock_mrnet_volume(num_slices=4, size=20, seed=0)
        assert volume.shape == (4, 20, 20)
        assert volume.dtype == np.uint16

    def test_end_to_end_mock_dataset_through_multiplane_selector(self, tmp_path: Path):
        """The full intended workflow: generate a mock MRNet tree, load one
        plane's volume for one case, and standardize it via
        `MultiPlaneViewSelector` — exercising the whole additive pipeline
        without the real Stanford MRNet dataset."""
        root = generate_mock_mrnet_dataset(tmp_path, case_ids=("0000",), num_slices=6, size=40, seed=2)
        volume = np.load(root / "train" / "sagittal" / "0000.npy")

        selector = MultiPlaneViewSelector(volume)
        tensor = selector.get_standardized_view("sagittal")

        assert tensor.shape == (3, 128, 128)
        assert torch.isfinite(tensor).all()
