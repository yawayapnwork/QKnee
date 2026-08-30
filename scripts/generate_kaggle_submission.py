"""
Generates an RSNA-Knee-format Kaggle `submission.csv` from a `test.csv` +
per-study multi-plane DICOM series directory tree, running real inference
through `qknee.models.pipeline.PipelineRunner`.

Expected `--series_dir` layout (see `qknee.data.dataset.RSNAKneeDataset`):

    <series_dir>/
        <StudyInstanceUID>/
            Sagittal/*.dcm
            Coronal/*.dcm
            Axial/*.dcm

Not every study needs every plane present.

MODEL COVERAGE — read before trusting these numbers:
    This repository's trained VQC heads currently cover only two of the
    12 RSNA target conditions — ACL and Meniscus (not split by
    compartment). This script therefore:

        - Runs REAL hybrid quantum inference (ResNet18 -> PCA -> VQC) for
          the "ACL" column, using the ACL-trained head.
        - Runs REAL hybrid quantum inference once for "Meniscus" (using
          the Meniscus-trained head) and uses that SAME score for both
          "Medial Meniscus" and "Lateral Meniscus" — no per-compartment
          head exists yet, so this is a documented stand-in, not two
          independent predictions.
        - Fills the remaining 9 columns (MCL, Medial OA, Lateral OA, PF OA,
          Effusion, Synovitis, Baker's, Contusion, Fracture) with a fixed,
          clearly-labeled `--default_probability` (default 0.1) — this is
          NOT a model prediction. Training/plugging in real heads for
          these conditions is required before this submission's scores
          for those 9 columns mean anything.

    Every generated `submission.csv` logs exactly which columns were
    model-derived vs. placeholder, so this is never silently ambiguous.

Per-study aggregation: a study's per-plane DICOM series (Sagittal/Coronal/
Axial, whichever exist) are each run through the pipeline independently
(each plane's multi-slice volume is averaged into one embedding by
`ResNet18FeatureExtractor.forward_volume`, exactly as `PipelineRunner`
does for any multi-slice input); the resulting per-plane scores are then
aggregated into one per-study score via `--aggregation` (`mean` (default)
or `max`). A study with zero available planes falls back to
`--default_probability` for its model-derived columns too (logged).

Usage:
    python scripts/generate_kaggle_submission.py --test_csv data/test.csv --series_dir data/test_series
    python scripts/generate_kaggle_submission.py --test_csv data/test.csv --series_dir data/test_series \\
        --aggregation max --default_probability 0.15 --output qknee/artifacts/submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow `python scripts/generate_kaggle_submission.py` to resolve the
# `qknee` package without requiring the caller to set PYTHONPATH or use
# `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.dataset import RSNA_PLANES, RSNA_TARGET_COLUMNS, RSNA_UID_COLUMN, RSNAKneeDataset, RSNAStudyRecord

logger = get_logger(__name__)
_config = load_config()

DEFAULT_OUTPUT_PATH = Path("qknee/artifacts/submission.csv")

# Which RSNA_TARGET_COLUMNS this repo's trained heads actually cover, and
# which head each maps to — everything else in RSNA_TARGET_COLUMNS falls
# back to --default_probability. Single source of truth for both the
# inference loop and the "which columns are real" log line below.
MODEL_COVERED_COLUMNS: Dict[str, str] = {
    "ACL": "acl",
    "Medial Meniscus": "meniscus",
    "Lateral Meniscus": "meniscus",
}
PLACEHOLDER_COLUMNS = tuple(c for c in RSNA_TARGET_COLUMNS if c not in MODEL_COVERED_COLUMNS)


class SubmissionValidationError(RuntimeError):
    """Raised when the generated submission fails schema/range/coverage validation."""


# --------------------------------------------------------------------------- #
# Backend loading (mirrors qknee.ui.dashboard.load_backend's pattern —
# reimplemented locally rather than imported, so this CLI script doesn't
# pull in Streamlit just to build a PipelineRunner + two VQC heads)
# --------------------------------------------------------------------------- #

def load_backend(device: torch.device):
    """Builds a `PipelineRunner` plus its ACL and Meniscus `VQCClassifier`
    heads, loading each head's trained checkpoint when available and
    falling back to randomly-initialized weights (with a loud warning)
    otherwise — same "trained if available, else honestly-labeled demo
    weights" convention every other entry point in this project follows."""
    from qknee.models.pipeline import PipelineRunner, PipelineValidationError, load_vqc_weights
    from qknee.models.vqc import VQCClassifier

    if not _config.paths.pca_artifact.exists():
        raise SubmissionValidationError(
            f"No fitted PCA artifact found at {_config.paths.pca_artifact}. Fit one first "
            "(see qknee.models.pca_reducer / scripts/train.py) before generating a submission."
        )

    runner = PipelineRunner(config=_config)

    torch.manual_seed(42)
    acl_model = VQCClassifier()
    if _config.paths.acl_checkpoint.exists():
        try:
            load_vqc_weights(acl_model, _config.paths.acl_checkpoint)
            logger.info("Loaded trained ACL VQC weights from %s", _config.paths.acl_checkpoint)
        except PipelineValidationError as exc:
            logger.warning("Failed to load ACL checkpoint; using random weights: %s", exc)
    else:
        logger.warning("No ACL checkpoint found at %s; using randomly initialized weights.", _config.paths.acl_checkpoint)
    acl_model.to(device).eval()

    torch.manual_seed(7)
    meniscus_model = VQCClassifier()
    if _config.paths.meniscus_checkpoint.exists():
        try:
            load_vqc_weights(meniscus_model, _config.paths.meniscus_checkpoint)
            logger.info("Loaded trained meniscus VQC weights from %s", _config.paths.meniscus_checkpoint)
        except PipelineValidationError as exc:
            logger.warning("Failed to load meniscus checkpoint; using random weights: %s", exc)
    else:
        logger.warning("No meniscus checkpoint found at %s; using randomly initialized weights.", _config.paths.meniscus_checkpoint)
    meniscus_model.to(device).eval()

    return runner, {"acl": acl_model, "meniscus": meniscus_model}


# --------------------------------------------------------------------------- #
# Per-study, per-plane inference + aggregation
# --------------------------------------------------------------------------- #

def score_plane(runner, vqc_head, plane_dir: Path) -> Optional[float]:
    """Runs one plane's DICOM series through the full hybrid pipeline
    (ingest -> ResNet18 -> PCA -> VQC) and returns its risk score, or
    `None` (logged) if that plane's series fails to load/infer — a single
    corrupted series shouldn't abort the whole study."""
    try:
        batch = runner.ingest(plane_dir)
        features = runner.extract_resnet_features(batch)
        quantum_angles = runner.reduce_to_quantum_angles(features)
        return runner.classify(quantum_angles, vqc=vqc_head)
    except Exception as exc:  # noqa: BLE001 - one bad series must not abort the whole submission
        logger.warning("Inference failed for plane series %s: %s", plane_dir, exc)
        return None


def aggregate_scores(scores: List[float], aggregation: str) -> Optional[float]:
    if not scores:
        return None
    if aggregation == "max":
        return float(max(scores))
    return float(np.mean(scores))  # "mean" (default)


def score_study(
    record: RSNAStudyRecord,
    runner,
    vqc_heads: Dict[str, object],
    aggregation: str,
    default_probability: float,
) -> Dict[str, float]:
    """Produces one study's full 12-column score dict: real hybrid
    quantum-model inference for `MODEL_COVERED_COLUMNS`, `default_probability`
    for everything else (and as the fallback for a model-covered column
    when the study has zero usable plane series)."""
    row: Dict[str, float] = {column: default_probability for column in RSNA_TARGET_COLUMNS}

    if not record.plane_series_dirs:
        logger.warning(
            "Study %s has no DICOM series available; all columns fall back to default_probability=%.3f.",
            record.study_instance_uid, default_probability,
        )
        return row

    # One inference pass per (head, plane) — the two heads share the same
    # ResNet18/PCA features per plane conceptually, but PipelineRunner's
    # public stage methods don't cache across calls, so this recomputes
    # features per head. Simplicity over micro-optimizing a CLI batch job.
    for head_name in set(MODEL_COVERED_COLUMNS.values()):
        vqc_head = vqc_heads[head_name]
        plane_scores = [
            score for plane_dir in record.plane_series_dirs.values()
            if (score := score_plane(runner, vqc_head, plane_dir)) is not None
        ]
        aggregated = aggregate_scores(plane_scores, aggregation)
        if aggregated is None:
            logger.warning(
                "Study %s: all plane series failed inference for the '%s' head; falling back to default_probability.",
                record.study_instance_uid, head_name,
            )
            aggregated = default_probability

        for column, mapped_head in MODEL_COVERED_COLUMNS.items():
            if mapped_head == head_name:
                row[column] = aggregated

    return row


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_submission(submission: pd.DataFrame, expected_uids: List[str]) -> None:
    """Asserts `submission` is a compliant RSNA-Knee submission: exact
    column headers (name and order), valid `[0.0, 1.0]` probabilities with
    no NaN, and a `StudyInstanceUID` set that matches `expected_uids`
    exactly (same count, no missing, no unexpected extras, no duplicates).

    Raises:
        SubmissionValidationError: naming exactly what failed.
    """
    expected_columns = [RSNA_UID_COLUMN, *RSNA_TARGET_COLUMNS]
    if list(submission.columns) != expected_columns:
        raise SubmissionValidationError(
            f"Column headers do not match the required schema.\nExpected: {expected_columns}\nGot:      {list(submission.columns)}"
        )

    if submission[RSNA_UID_COLUMN].isna().any():
        raise SubmissionValidationError("submission.csv has one or more null StudyInstanceUID values.")

    duplicated = submission[RSNA_UID_COLUMN][submission[RSNA_UID_COLUMN].duplicated()].tolist()
    if duplicated:
        raise SubmissionValidationError(f"submission.csv has duplicate StudyInstanceUID value(s): {duplicated[:10]}")

    submitted_uids = set(submission[RSNA_UID_COLUMN])
    expected_uid_set = set(expected_uids)
    missing = expected_uid_set - submitted_uids
    extra = submitted_uids - expected_uid_set
    if missing or extra:
        raise SubmissionValidationError(
            f"StudyInstanceUID mismatch between submission.csv and the input test set "
            f"(submission has {len(submitted_uids)}, expected {len(expected_uid_set)}). "
            f"Missing: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}; "
            f"Unexpected: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
        )

    for column in RSNA_TARGET_COLUMNS:
        values = submission[column]
        if values.isna().any():
            raise SubmissionValidationError(f"Column '{column}' has {int(values.isna().sum())} NaN value(s) — a submission cannot contain NaN.")
        if not pd.api.types.is_numeric_dtype(values):
            raise SubmissionValidationError(f"Column '{column}' is not numeric (dtype={values.dtype}).")
        out_of_range = values[(values < 0.0) | (values > 1.0)]
        if not out_of_range.empty:
            raise SubmissionValidationError(
                f"Column '{column}' has {len(out_of_range)} value(s) outside [0.0, 1.0], "
                f"e.g. {out_of_range.tolist()[:5]}."
            )

    logger.info(
        "Validation passed: %d rows, %d columns, schema/range/UID-count all OK.",
        len(submission), len(submission.columns),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_submission(
    test_csv: Path,
    series_dir: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    aggregation: str = "mean",
    default_probability: float = 0.1,
    device: Optional[str] = None,
    limit: Optional[int] = None,
) -> Path:
    device_obj = torch.device(device or _config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    logger.info(
        "Model-derived columns: %s | Placeholder columns (default_probability=%.3f): %s",
        sorted(set(MODEL_COVERED_COLUMNS.keys())), default_probability, list(PLACEHOLDER_COLUMNS),
    )

    dataset = RSNAKneeDataset(test_csv, series_dir, planes=RSNA_PLANES, require_targets=False)
    records = list(dataset)
    if limit is not None:
        records = records[:limit]
        logger.info("--limit=%d: scoring only the first %d of %d studies.", limit, len(records), len(dataset))

    expected_uids = [record.study_instance_uid for record in records]
    logger.info("Loaded %d studies from %s (series root: %s)", len(records), test_csv, series_dir)

    runner, vqc_heads = load_backend(device_obj)

    rows = []
    for i, record in enumerate(records):
        row = {RSNA_UID_COLUMN: record.study_instance_uid}
        row.update(score_study(record, runner, vqc_heads, aggregation, default_probability))
        rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(records):
            logger.info("Scored %d/%d studies...", i + 1, len(records))

    submission = pd.DataFrame(rows, columns=[RSNA_UID_COLUMN, *RSNA_TARGET_COLUMNS])

    validate_submission(submission, expected_uids)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    logger.info("Wrote %s (%d rows)", output_path, len(submission))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_csv", type=Path, required=True, help="Path to the RSNA-Knee-format test.csv.")
    parser.add_argument("--series_dir", type=Path, required=True, help="Root directory: <series_dir>/<StudyInstanceUID>/<Plane>/*.dcm")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination submission.csv path.")
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean", help="Cross-plane score aggregation per study.")
    parser.add_argument("--default_probability", type=float, default=0.1, help="Fallback probability for placeholder/unmodeled columns and studies with no usable series.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string; defaults to config.yaml's device or CUDA-if-available.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N studies (smoke-testing).")
    args = parser.parse_args()

    setup_logging()

    if not 0.0 <= args.default_probability <= 1.0:
        raise SubmissionValidationError(f"--default_probability must be in [0.0, 1.0], got {args.default_probability}")

    generate_submission(
        test_csv=args.test_csv,
        series_dir=args.series_dir,
        output_path=args.output,
        aggregation=args.aggregation,
        default_probability=args.default_probability,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    try:
        main()
    except SubmissionValidationError as exc:
        logger.error("Submission generation aborted: %s", exc)
        sys.exit(1)
