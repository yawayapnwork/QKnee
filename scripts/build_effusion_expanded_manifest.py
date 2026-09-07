"""
Builds the provenance-tagged Effusion training manifest: every ground-truth
study (58, from ../rsna-knee/train_series) and every confidently
weak-labeled study (qknee/artifacts/effusion_scaled_labels.csv, no
`needs_review` flag) in ONE table, each row explicitly tagged
`provenance in {"ground_truth", "weak_labeled"}` -- the two pools are never
merged into an indistinguishable label column. Also records whether each
weak-labeled study's DICOM images are present on disk yet (the download in
scripts/_fetch_rsna_effusion_weak_labeled.py runs separately and
incrementally), since only image-available rows are usable for training.

Ground-truth Effusion labels are read from train.csv directly (not
train_series.csv); weak-labeled labels are `assigned_label` from
effusion_scaled_labels.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_SERIES_DIR = REPO_ROOT.parent / "rsna-knee" / "train_series"
WEAK_SERIES_DIR = REPO_ROOT / "train_series"
LABELS_CSV = REPO_ROOT / "qknee" / "artifacts" / "effusion_scaled_labels.csv"
TRAIN_CSV = REPO_ROOT / "train.csv"
OUTPUT_CSV = REPO_ROOT / "qknee" / "artifacts" / "effusion_expanded_manifest.csv"


def _has_images(series_dir: Path, study_uid: str) -> bool:
    study_dir = series_dir / study_uid
    return study_dir.is_dir() and any(study_dir.rglob("*.dcm"))


def main() -> None:
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    gt = train.loc[train["Effusion"].notna(), ["StudyInstanceUID", "Effusion"]].copy()
    gt["provenance"] = "ground_truth"
    gt["tier"] = "GROUND_TRUTH"
    gt = gt.rename(columns={"Effusion": "effusion_label"})
    gt["effusion_label"] = gt["effusion_label"].astype(int)
    gt["images_available"] = gt["StudyInstanceUID"].apply(lambda u: _has_images(GT_SERIES_DIR, u))

    weak = pd.read_csv(LABELS_CSV, dtype={"StudyInstanceUID": str})
    confident = weak[weak["needs_review"].isna() | (weak["needs_review"].astype(str).str.strip() == "")].copy()
    confident["provenance"] = "weak_labeled"
    confident = confident.rename(columns={"assigned_label": "effusion_label"})[
        ["StudyInstanceUID", "effusion_label", "provenance", "tier"]
    ]
    confident["images_available"] = confident["StudyInstanceUID"].apply(lambda u: _has_images(WEAK_SERIES_DIR, u))

    combined = pd.concat(
        [gt[["StudyInstanceUID", "effusion_label", "provenance", "tier", "images_available"]], confident],
        ignore_index=True,
    )
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"ground_truth: {len(gt)} studies, {gt['images_available'].sum()} with images on disk")
    print(f"weak_labeled: {len(confident)} studies, {confident['images_available'].sum()} with images on disk")
    print(f"wrote {OUTPUT_CSV} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
