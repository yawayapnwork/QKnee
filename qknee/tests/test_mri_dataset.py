"""
Tests for `qknee.data.dataset.MRIDataset`, `build_dataloaders`, and
`collate_skip_invalid`. Covers:

    1. Directory discovery / class indexing for both image (.png) and
       volume (.npy) files, including `labeled=False` mode.
    2. Corrupted-file handling at two points: files that are already
       corrupted at index-build time (skipped, never indexed) and files
       that become corrupted *after* indexing (surfaced as `None` from
       `__getitem__`, not a mislabeled placeholder tensor).
    3. `collate_skip_invalid` dropping `None` (invalid) samples from a
       batch cleanly, including the all-invalid-batch edge case, wired
       through a real `DataLoader` via `build_dataloaders`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from qknee.data.dataset import MRIDataset, build_dataloaders, build_transforms, collate_skip_invalid


def _write_png(path: Path, seed: int, size: int = 32) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def _write_npy_volume(path: Path, seed: int, depth: int = 4, size: int = 32) -> None:
    rng = np.random.default_rng(seed)
    volume = rng.integers(0, 255, size=(depth, size, size)).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, volume)


def _corrupt(path: Path) -> None:
    """Overwrites an existing file with garbage bytes, simulating
    corruption that happens *after* a dataset has already indexed it."""
    path.write_bytes(b"not a valid image or array")


# --------------------------------------------------------------------------- #
# 1. Directory discovery / class indexing
# --------------------------------------------------------------------------- #

class TestMRIDatasetIndexing:
    def test_indexes_labeled_class_subdirectories(self, tmp_path: Path):
        _write_png(tmp_path / "no_tear" / "a.png", seed=1)
        _write_png(tmp_path / "no_tear" / "b.png", seed=2)
        _write_png(tmp_path / "tear" / "c.png", seed=3)

        dataset = MRIDataset(tmp_path)

        assert sorted(dataset.classes) == ["no_tear", "tear"]
        assert len(dataset) == 3
        labels = {dataset[i][1] for i in range(len(dataset))}
        assert labels == {dataset.class_to_idx["no_tear"], dataset.class_to_idx["tear"]}

    def test_unlabeled_mode_assigns_sentinel_label(self, tmp_path: Path):
        _write_png(tmp_path / "a.png", seed=1)
        _write_png(tmp_path / "b.png", seed=2)

        dataset = MRIDataset(tmp_path, labeled=False)

        assert len(dataset) == 2
        assert all(dataset[i][1] == -1 for i in range(len(dataset)))

    def test_npy_volume_expands_into_one_sample_per_slice(self, tmp_path: Path):
        _write_npy_volume(tmp_path / "no_tear" / "volume.npy", seed=1, depth=5)

        dataset = MRIDataset(tmp_path, transform=build_transforms(train=False))

        assert len(dataset) == 5
        tensor, label = dataset[0]
        assert tensor.shape == (3, 224, 224)  # after build_transforms' Resize + Grayscale(3)

    def test_missing_root_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            MRIDataset(tmp_path / "does_not_exist")

    def test_empty_directory_raises_runtime_error(self, tmp_path: Path):
        (tmp_path / "no_tear").mkdir()
        with pytest.raises(RuntimeError, match="No valid samples"):
            MRIDataset(tmp_path)

    def test_labeled_without_class_subdirectories_raises(self, tmp_path: Path):
        _write_png(tmp_path / "loose.png", seed=1)  # no class subfolder
        with pytest.raises(RuntimeError, match="no class subdirectories"):
            MRIDataset(tmp_path)


# --------------------------------------------------------------------------- #
# 2. Corrupted-file handling
# --------------------------------------------------------------------------- #

class TestCorruptedFileHandling:
    def test_corrupted_image_is_skipped_at_index_time(self, tmp_path: Path):
        _write_png(tmp_path / "no_tear" / "good.png", seed=1)
        corrupt_path = tmp_path / "no_tear" / "bad.png"
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_bytes(b"not a real png")

        dataset = MRIDataset(tmp_path)

        # Only the good file was indexed; the corrupted one never entered `samples`.
        assert len(dataset) == 1
        assert dataset.samples[0].path.name == "good.png"

    def test_corrupted_volume_is_skipped_at_index_time(self, tmp_path: Path):
        _write_png(tmp_path / "no_tear" / "good.png", seed=1)
        corrupt_path = tmp_path / "no_tear" / "bad.npy"
        corrupt_path.parent.mkdir(parents=True, exist_ok=True)
        corrupt_path.write_bytes(b"not a real npy file")

        dataset = MRIDataset(tmp_path)

        assert len(dataset) == 1  # the corrupted volume contributed zero samples

    def test_wrong_ndim_volume_is_skipped_at_index_time(self, tmp_path: Path):
        """`.npy` files are always treated as (D, H, W[, C]) volumes; a
        plain 2D array doesn't satisfy that and is skipped, even though the
        same shape would be a perfectly valid standalone image file."""
        _write_png(tmp_path / "no_tear" / "good.png", seed=1)
        rng = np.random.default_rng(0)
        np.save(tmp_path / "no_tear" / "flat.npy", rng.integers(0, 255, size=(32, 32)).astype(np.uint8))

        dataset = MRIDataset(tmp_path)

        assert len(dataset) == 1

    def test_getitem_returns_none_for_file_corrupted_after_indexing(self, tmp_path: Path):
        """A file that was valid when the dataset indexed it, but becomes
        unreadable afterward (e.g. truncated by a concurrent write), must
        surface as `None` from `__getitem__` — not a fabricated placeholder
        tensor under a real label."""
        image_path = tmp_path / "no_tear" / "will_break.png"
        _write_png(image_path, seed=1)

        dataset = MRIDataset(tmp_path)
        assert len(dataset) == 1  # indexed successfully while still valid

        _corrupt(image_path)

        result = dataset[0]
        assert result is None

    def test_getitem_succeeds_normally_for_valid_samples(self, tmp_path: Path):
        _write_png(tmp_path / "no_tear" / "good.png", seed=1)
        dataset = MRIDataset(tmp_path, transform=build_transforms(train=False))

        result = dataset[0]
        assert result is not None
        tensor, label = result
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert label == dataset.class_to_idx["no_tear"]


# --------------------------------------------------------------------------- #
# 3. collate_skip_invalid
# --------------------------------------------------------------------------- #

class TestCollateSkipInvalid:
    def test_drops_none_entries_and_batches_the_rest(self):
        good_a = (torch.rand(3, 224, 224), 0)
        good_b = (torch.rand(3, 224, 224), 1)
        batch = [good_a, None, good_b]

        collated = collate_skip_invalid(batch)

        assert collated is not None
        images, labels = collated
        assert images.shape == (2, 3, 224, 224)
        assert labels.tolist() == [0, 1]

    def test_returns_none_when_entire_batch_is_invalid(self):
        batch = [None, None, None]
        assert collate_skip_invalid(batch) is None

    def test_no_invalid_entries_behaves_like_default_collate(self):
        batch = [(torch.rand(3, 224, 224), 0), (torch.rand(3, 224, 224), 1)]
        collated = collate_skip_invalid(batch)
        assert collated is not None
        images, labels = collated
        assert images.shape == (2, 3, 224, 224)

    def test_build_dataloaders_end_to_end_with_one_corrupted_file(self, tmp_path: Path):
        """A DataLoader built via `build_dataloaders` uses
        `collate_skip_invalid`, so a file corrupted after indexing is
        dropped from its batch instead of crashing iteration or injecting a
        mislabeled placeholder."""
        train_dir = tmp_path / "train"
        for i in range(6):
            _write_png(train_dir / "no_tear" / f"img_{i}.png", seed=i)
        breaking_path = train_dir / "no_tear" / "img_0.png"

        loaders = build_dataloaders(tmp_path, batch_size=6, num_workers=0, labeled=True)
        assert "train" in loaders

        # Corrupt one already-indexed file before iterating.
        _corrupt(breaking_path)

        batches = [batch for batch in loaders["train"] if batch is not None]
        assert len(batches) == 1
        images, labels = batches[0]
        # 6 samples indexed, 1 corrupted post-index -> 5 survive collation.
        assert images.shape[0] == 5
