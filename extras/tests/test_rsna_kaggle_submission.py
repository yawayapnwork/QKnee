"""
Tests for the RSNA Knee dataset parser (`qknee.data.dataset`) and the
Kaggle submission exporter (`scripts.generate_kaggle_submission`).

Covers:
    1. `load_rsna_labels_csv`: robust null-handling (missing/empty UID
       rows dropped, missing target columns filled with NaN, non-numeric
       cells coerced to NaN, out-of-range/duplicate values logged but not
       fatal), and `require_targets` enforcement.
    2. `discover_rsna_plane_series` / `RSNAKneeDataset`: partial-plane
       studies, studies with zero series, and the "one record per CSV
       row, never dropped for a missing series" guarantee.
    3. `validate_submission`: every failure mode (schema, range, NaN,
       UID mismatch, duplicates) and the valid-submission pass-through.
    4. `generate_submission` end-to-end against a synthetic mock DICOM
       test set, including the zero-series fallback path — asserting the
       final CSV's schema, UID count, and probability ranges directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qknee.data.dataset import (
    RSNA_PLANES,
    RSNA_TARGET_COLUMNS,
    RSNA_UID_COLUMN,
    RSNAKneeDataset,
    discover_rsna_plane_series,
    load_rsna_labels_csv,
)
from qknee.data.ingestion import generate_mock_dicom_series

pytestmark = [pytest.mark.slow]  # generate_submission tests build a real PipelineRunner


def _write_csv(path: Path, rows: list) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _full_target_row(uid: str, **overrides) -> dict:
    row = {RSNA_UID_COLUMN: uid, **{col: 0 for col in RSNA_TARGET_COLUMNS}}
    row.update(overrides)
    return row


# --------------------------------------------------------------------------- #
# 1. load_rsna_labels_csv
# --------------------------------------------------------------------------- #

class TestLoadRsnaLabelsCsv:
    def test_missing_uid_column_raises(self, tmp_path: Path):
        csv_path = _write_csv(tmp_path / "bad.csv", [{"NotTheRightColumn": "x"}])
        with pytest.raises(ValueError, match="StudyInstanceUID"):
            load_rsna_labels_csv(csv_path)

    def test_missing_target_columns_filled_with_nan(self, tmp_path: Path):
        csv_path = _write_csv(tmp_path / "test.csv", [{RSNA_UID_COLUMN: "uid-1"}])
        df = load_rsna_labels_csv(csv_path, require_targets=False)
        assert set(RSNA_TARGET_COLUMNS).issubset(df.columns)
        assert df["ACL"].isna().all()

    def test_require_targets_raises_when_columns_absent(self, tmp_path: Path):
        csv_path = _write_csv(tmp_path / "test.csv", [{RSNA_UID_COLUMN: "uid-1"}])
        with pytest.raises(ValueError, match="require_targets"):
            load_rsna_labels_csv(csv_path, require_targets=True)

    def test_rows_with_missing_or_empty_uid_are_dropped(self, tmp_path: Path):
        rows = [
            _full_target_row("uid-1"),
            {RSNA_UID_COLUMN: None, **{c: 0 for c in RSNA_TARGET_COLUMNS}},
            {RSNA_UID_COLUMN: "  ", **{c: 0 for c in RSNA_TARGET_COLUMNS}},
        ]
        csv_path = _write_csv(tmp_path / "train.csv", rows)
        df = load_rsna_labels_csv(csv_path, require_targets=True)
        assert len(df) == 1
        assert df[RSNA_UID_COLUMN].tolist() == ["uid-1"]

    def test_none_target_value_becomes_nan(self, tmp_path: Path):
        csv_path = _write_csv(tmp_path / "train.csv", [_full_target_row("uid-1", MCL=None)])
        df = load_rsna_labels_csv(csv_path, require_targets=True)
        assert pd.isna(df.loc[0, "MCL"])

    def test_non_numeric_cell_coerced_to_nan_not_raised(self, tmp_path: Path):
        csv_path = _write_csv(tmp_path / "train.csv", [_full_target_row("uid-1", Fracture="not-a-number")])
        df = load_rsna_labels_csv(csv_path, require_targets=True)
        assert pd.isna(df.loc[0, "Fracture"])

    def test_out_of_range_and_duplicate_values_do_not_raise(self, tmp_path: Path, caplog):
        rows = [
            _full_target_row("uid-1", ACL=1.5),
            _full_target_row("uid-1"),  # duplicate UID
        ]
        csv_path = _write_csv(tmp_path / "train.csv", rows)
        df = load_rsna_labels_csv(csv_path, require_targets=True)  # must not raise
        assert len(df) == 2

    def test_missing_csv_file_raises_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_rsna_labels_csv(tmp_path / "does_not_exist.csv")


# --------------------------------------------------------------------------- #
# 2. discover_rsna_plane_series / RSNAKneeDataset
# --------------------------------------------------------------------------- #

class TestRSNAKneeDataset:
    @pytest.fixture
    def series_root(self, tmp_path: Path) -> Path:
        root = tmp_path / "series"
        generate_mock_dicom_series(root / "uid-full" / "Sagittal", num_slices=2, rows=16, columns=16, seed=1)
        generate_mock_dicom_series(root / "uid-full" / "Coronal", num_slices=2, rows=16, columns=16, seed=2)
        generate_mock_dicom_series(root / "uid-full" / "Axial", num_slices=2, rows=16, columns=16, seed=3)
        generate_mock_dicom_series(root / "uid-partial" / "Sagittal", num_slices=2, rows=16, columns=16, seed=4)
        return root

    def test_discover_finds_only_planes_with_dicom_files(self, series_root: Path):
        found = discover_rsna_plane_series(series_root, "uid-full")
        assert set(found.keys()) == set(RSNA_PLANES)

    def test_discover_partial_study(self, series_root: Path):
        found = discover_rsna_plane_series(series_root, "uid-partial")
        assert set(found.keys()) == {"Sagittal"}

    def test_discover_missing_study_returns_empty_dict(self, series_root: Path):
        assert discover_rsna_plane_series(series_root, "no-such-study") == {}

    def test_every_csv_row_becomes_one_record_even_without_series(self, tmp_path: Path, series_root: Path):
        csv_path = _write_csv(tmp_path / "train.csv", [
            _full_target_row("uid-full"),
            _full_target_row("uid-partial"),
            _full_target_row("uid-with-no-series-at-all"),
        ])
        dataset = RSNAKneeDataset(csv_path, series_root, require_targets=True)

        assert len(dataset) == 3
        by_uid = {r.study_instance_uid: r for r in dataset}
        assert set(by_uid["uid-full"].plane_series_dirs.keys()) == set(RSNA_PLANES)
        assert set(by_uid["uid-partial"].plane_series_dirs.keys()) == {"Sagittal"}
        assert by_uid["uid-with-no-series-at-all"].plane_series_dirs == {}

    def test_targets_are_none_for_test_csv_without_target_columns(self, tmp_path: Path, series_root: Path):
        csv_path = _write_csv(tmp_path / "test.csv", [{RSNA_UID_COLUMN: "uid-full"}])
        dataset = RSNAKneeDataset(csv_path, series_root, require_targets=False)
        assert dataset[0].targets is None

    def test_targets_dict_present_for_labeled_train_csv(self, tmp_path: Path, series_root: Path):
        csv_path = _write_csv(tmp_path / "train.csv", [_full_target_row("uid-full", ACL=1)])
        dataset = RSNAKneeDataset(csv_path, series_root, require_targets=True)
        assert dataset[0].targets is not None
        assert dataset[0].targets["ACL"] == 1.0
        assert set(dataset[0].targets.keys()) == set(RSNA_TARGET_COLUMNS)

    def test_dataset_is_indexable_and_iterable(self, tmp_path: Path, series_root: Path):
        csv_path = _write_csv(tmp_path / "test.csv", [
            {RSNA_UID_COLUMN: "uid-full"}, {RSNA_UID_COLUMN: "uid-partial"},
        ])
        dataset = RSNAKneeDataset(csv_path, series_root)
        assert dataset[0].study_instance_uid == "uid-full"
        assert [r.study_instance_uid for r in dataset] == ["uid-full", "uid-partial"]


# --------------------------------------------------------------------------- #
# 3. validate_submission
# --------------------------------------------------------------------------- #

class TestValidateSubmission:
    @pytest.fixture(autouse=True)
    def _import_validator(self):
        from scripts.generate_kaggle_submission import SubmissionValidationError, validate_submission
        self.validate_submission = validate_submission
        self.SubmissionValidationError = SubmissionValidationError

    def _valid_frame(self, uids):
        rows = [{RSNA_UID_COLUMN: uid, **{c: 0.5 for c in RSNA_TARGET_COLUMNS}} for uid in uids]
        return pd.DataFrame(rows, columns=[RSNA_UID_COLUMN, *RSNA_TARGET_COLUMNS])

    def test_valid_submission_passes(self):
        df = self._valid_frame(["a", "b"])
        self.validate_submission(df, ["a", "b"])  # must not raise

    def test_wrong_column_order_raises(self):
        df = self._valid_frame(["a"])[[RSNA_UID_COLUMN, *reversed(RSNA_TARGET_COLUMNS)]]
        with pytest.raises(self.SubmissionValidationError, match="Column headers"):
            self.validate_submission(df, ["a"])

    def test_out_of_range_value_raises(self):
        df = self._valid_frame(["a"])
        df.loc[0, "ACL"] = 1.5
        with pytest.raises(self.SubmissionValidationError, match="outside \\[0.0, 1.0\\]"):
            self.validate_submission(df, ["a"])

    def test_negative_value_raises(self):
        df = self._valid_frame(["a"])
        df.loc[0, "Fracture"] = -0.1
        with pytest.raises(self.SubmissionValidationError, match="outside \\[0.0, 1.0\\]"):
            self.validate_submission(df, ["a"])

    def test_nan_value_raises(self):
        df = self._valid_frame(["a"])
        df.loc[0, "MCL"] = np.nan
        with pytest.raises(self.SubmissionValidationError, match="NaN"):
            self.validate_submission(df, ["a"])

    def test_missing_uid_raises(self):
        df = self._valid_frame(["a"])
        with pytest.raises(self.SubmissionValidationError, match="mismatch"):
            self.validate_submission(df, ["a", "b"])

    def test_unexpected_extra_uid_raises(self):
        df = self._valid_frame(["a", "b"])
        with pytest.raises(self.SubmissionValidationError, match="mismatch"):
            self.validate_submission(df, ["a"])

    def test_duplicate_uid_raises(self):
        df = self._valid_frame(["a", "a"])
        with pytest.raises(self.SubmissionValidationError, match="duplicate"):
            self.validate_submission(df, ["a"])


# --------------------------------------------------------------------------- #
# 4. generate_submission end-to-end
# --------------------------------------------------------------------------- #

class TestGenerateSubmissionEndToEnd:
    @pytest.fixture
    def synthetic_test_set(self, tmp_path: Path, pca_artifact_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A small synthetic RSNA-shaped test.csv + mock DICOM series tree,
        with the PCA artifact path patched to a real fitted one (via the
        shared `pca_artifact_path` fixture) so `generate_submission`'s
        `PipelineRunner` construction succeeds without a real dataset."""
        from qknee.config import loader as config_loader

        series_dir = tmp_path / "test_series"
        uids = ["uid-0", "uid-1", "uid-no-series"]
        for i, uid in enumerate(uids[:2]):
            for plane in RSNA_PLANES:
                generate_mock_dicom_series(
                    series_dir / uid / plane, num_slices=2, rows=32, columns=32, seed=i * 3 + hash(plane) % 5,
                )
        csv_path = _write_csv(tmp_path / "test.csv", [{RSNA_UID_COLUMN: uid} for uid in uids])

        # Patch the module-level `_config` object `generate_kaggle_submission`
        # imported at module load time, so it points at the test's fitted
        # PCA artifact instead of the real (possibly-missing) one.
        import scripts.generate_kaggle_submission as submission_module
        import dataclasses
        patched_config = dataclasses.replace(
            submission_module._config,
            paths=dataclasses.replace(submission_module._config.paths, pca_artifact=pca_artifact_path),
        )
        monkeypatch.setattr(submission_module, "_config", patched_config)

        return csv_path, series_dir, uids

    def _write_constant_train_csv(self, tmp_path: Path, value: float, name: str = "train_priors.csv") -> Path:
        """A train.csv whose every row has the same value for every target
        column, so `compute_class_priors` (the column mean) resolves to
        exactly `value` — lets tests assert an exact class-prior fallback
        without depending on FLAT_CLASS_PRIOR."""
        rows = [_full_target_row(f"prior-uid-{i}", **{c: value for c in RSNA_TARGET_COLUMNS}) for i in range(3)]
        return _write_csv(tmp_path / name, rows)

    def test_generate_submission_produces_valid_csv(self, synthetic_test_set, tmp_path: Path):
        from scripts.generate_kaggle_submission import generate_submission

        csv_path, series_dir, uids = synthetic_test_set
        output_path = tmp_path / "submission.csv"

        result_path = generate_submission(
            input_csv=csv_path, images_dir=series_dir, output_csv=output_path, aggregation="mean", num_workers=0,
        )

        assert result_path == output_path
        assert output_path.exists()

        submission = pd.read_csv(output_path)
        assert list(submission.columns) == [RSNA_UID_COLUMN, *RSNA_TARGET_COLUMNS]
        assert len(submission) == len(uids)
        assert set(submission[RSNA_UID_COLUMN]) == set(uids)
        for column in RSNA_TARGET_COLUMNS:
            assert submission[column].notna().all()
            assert (submission[column] >= 0.0).all() and (submission[column] <= 1.0).all()

    def test_study_with_no_series_falls_back_to_class_prior(self, synthetic_test_set, tmp_path: Path):
        from scripts.generate_kaggle_submission import generate_submission

        csv_path, series_dir, uids = synthetic_test_set
        output_path = tmp_path / "submission.csv"
        prior_value = 0.37
        train_csv = self._write_constant_train_csv(tmp_path, prior_value)

        generate_submission(
            input_csv=csv_path, images_dir=series_dir, output_csv=output_path, train_csv=train_csv, num_workers=0,
        )

        submission = pd.read_csv(output_path)
        no_series_row = submission[submission[RSNA_UID_COLUMN] == "uid-no-series"].iloc[0]
        for column in RSNA_TARGET_COLUMNS:
            assert no_series_row[column] == pytest.approx(prior_value)

    def test_placeholder_columns_equal_class_prior(self, synthetic_test_set, tmp_path: Path):
        from scripts.generate_kaggle_submission import PLACEHOLDER_COLUMNS, generate_submission

        csv_path, series_dir, uids = synthetic_test_set
        output_path = tmp_path / "submission.csv"
        prior_value = 0.42
        train_csv = self._write_constant_train_csv(tmp_path, prior_value)

        generate_submission(
            input_csv=csv_path, images_dir=series_dir, output_csv=output_path, train_csv=train_csv, num_workers=0,
        )

        submission = pd.read_csv(output_path)
        for column in PLACEHOLDER_COLUMNS:
            assert submission[column].tolist() == pytest.approx([prior_value] * len(submission))

    def test_medial_and_lateral_meniscus_share_the_same_score(self, synthetic_test_set, tmp_path: Path):
        from scripts.generate_kaggle_submission import generate_submission

        csv_path, series_dir, uids = synthetic_test_set
        output_path = tmp_path / "submission.csv"

        generate_submission(input_csv=csv_path, images_dir=series_dir, output_csv=output_path, num_workers=0)

        submission = pd.read_csv(output_path)
        pd.testing.assert_series_equal(
            submission["Medial Meniscus"], submission["Lateral Meniscus"], check_names=False,
        )

    def test_limit_scores_only_the_first_n_studies(self, synthetic_test_set, tmp_path: Path):
        from scripts.generate_kaggle_submission import generate_submission

        csv_path, series_dir, uids = synthetic_test_set
        output_path = tmp_path / "submission.csv"

        generate_submission(input_csv=csv_path, images_dir=series_dir, output_csv=output_path, limit=1, num_workers=0)

        submission = pd.read_csv(output_path)
        assert len(submission) == 1
        assert submission.iloc[0][RSNA_UID_COLUMN] == uids[0]

    def test_compress_flag_writes_gzip_file(self, synthetic_test_set, tmp_path: Path):
        from scripts.generate_kaggle_submission import generate_submission

        csv_path, series_dir, uids = synthetic_test_set
        output_path = tmp_path / "submission.csv"

        result_path = generate_submission(
            input_csv=csv_path, images_dir=series_dir, output_csv=output_path, compress=True, limit=1, num_workers=0,
        )

        assert result_path.suffix == ".gz"
        assert result_path.exists()
        submission = pd.read_csv(result_path, compression="gzip")
        assert list(submission.columns) == [RSNA_UID_COLUMN, *RSNA_TARGET_COLUMNS]

    def test_batch_size_and_num_workers_do_not_change_results(self, synthetic_test_set, tmp_path: Path):
        """DataLoader batching/prefetching is purely an execution-strategy
        change — the scored values for a given (deterministic seed) run
        should be identical regardless of --batch_size/--num_workers."""
        from scripts.generate_kaggle_submission import generate_submission

        csv_path, series_dir, uids = synthetic_test_set

        output_a = tmp_path / "a.csv"
        generate_submission(
            input_csv=csv_path, images_dir=series_dir, output_csv=output_a, batch_size=1, num_workers=0,
        )
        output_b = tmp_path / "b.csv"
        generate_submission(
            input_csv=csv_path, images_dir=series_dir, output_csv=output_b, batch_size=8, num_workers=0,
        )

        submission_a = pd.read_csv(output_a).sort_values(RSNA_UID_COLUMN).reset_index(drop=True)
        submission_b = pd.read_csv(output_b).sort_values(RSNA_UID_COLUMN).reset_index(drop=True)
        pd.testing.assert_frame_equal(submission_a, submission_b)
