"""
Generates an RSNA-Knee-format Kaggle `submission.csv` from a `test.csv` +
per-study multi-plane DICOM series directory tree, running real batch
inference through either `qknee.models.pipeline.PipelineRunner` (native
PyTorch/PennyLane, default) or `HybridONNXInferenceEngine` (`--backend onnx`)
for the classical (ResNet18 -> PCA) stage.

Expected `--images_dir` layout (see `qknee.data.dataset.RSNAKneeDataset`):

    <images_dir>/
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
          Effusion, Synovitis, Baker's, Contusion, Fracture) with that
          column's CLASS PRIOR (its label prevalence, computed from
          `--train_csv` if given, else a clearly-logged flat fallback) —
          this is NOT a model prediction. Training/plugging in real heads
          for these conditions is required before this submission's scores
          for those 9 columns mean anything.
        - A model-covered column also falls back to its class prior for
          any study whose plane series all fail to load/infer.

    Every generated `submission.csv` logs exactly which columns were
    model-derived vs. class-prior placeholders, so this is never silently
    ambiguous.

Execution pipeline:
    - `RSNAInferenceDataset` performs the actual per-study DICOM decode +
      ImageNet-normalize preprocessing in `__getitem__` — the expensive,
      CPU-bound disk-I/O step — so a `torch.utils.data.DataLoader` with
      `num_workers` worker processes and `pin_memory=True` prefetches it
      in parallel while the main process runs model inference.
    - Per-study aggregation: a study's per-plane volumes (Sagittal/
      Coronal/Axial, whichever exist) are each run through the classical
      + quantum stages independently, then aggregated into one per-study
      score via `--aggregation` (`mean` (default) or `max`). A study with
      zero available planes falls back to class priors entirely (logged).

Usage:
    python scripts/generate_kaggle_submission.py \\
        --input_csv data/test.csv --images_dir data/test_series --output_csv qknee/artifacts/submission.csv

    python scripts/generate_kaggle_submission.py \\
        --input_csv data/test.csv --images_dir data/test_series --train_csv data/train.csv \\
        --backend onnx --batch_size 16 --compress
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

# Allow `python scripts/generate_kaggle_submission.py` to resolve the
# `qknee` package without requiring the caller to set PYTHONPATH or use
# `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.dataset import (
    RSNA_PLANES,
    RSNA_TARGET_COLUMNS,
    RSNA_UID_COLUMN,
    RSNAKneeDataset,
    RSNAStudyRecord,
    load_rsna_labels_csv,
)
from qknee.data.ingestion import DataIngestion, IngestionError

logger = get_logger(__name__)
_config = load_config()

DEFAULT_OUTPUT_PATH = Path("qknee/artifacts/submission.csv")

# A flat prior used only when --train_csv isn't given to compute real,
# per-column label prevalences — always logged loudly so it's never
# mistaken for a measured value.
FLAT_CLASS_PRIOR = 0.1

# Which RSNA_TARGET_COLUMNS this repo's trained heads actually cover, and
# which head each maps to — everything else in RSNA_TARGET_COLUMNS falls
# back to its class prior. Single source of truth for both the inference
# loop and the "which columns are real" log line below.
MODEL_COVERED_COLUMNS: Dict[str, str] = {
    "ACL": "acl",
    "Medial Meniscus": "meniscus",
    "Lateral Meniscus": "meniscus",
}
PLACEHOLDER_COLUMNS = tuple(c for c in RSNA_TARGET_COLUMNS if c not in MODEL_COVERED_COLUMNS)


class SubmissionValidationError(RuntimeError):
    """Raised when the generated submission fails schema/range/coverage validation."""


# --------------------------------------------------------------------------- #
# Class priors (replace a single flat --default_probability with real,
# per-column label prevalence when a train.csv is available)
# --------------------------------------------------------------------------- #

def compute_class_priors(train_csv: Optional[Path]) -> Dict[str, float]:
    """Returns `{column: prior_probability}` for every `RSNA_TARGET_COLUMNS`
    entry — the fallback value used both for placeholder columns and for
    any study/column whose model inference fails.

    Without `--train_csv`, every column falls back to `FLAT_CLASS_PRIOR`
    (loudly logged, never silently assumed to be a real prevalence). With
    `--train_csv`, each column's prior is that column's mean label value
    across all non-null rows — a real, measured class prior.
    """
    if train_csv is None:
        logger.warning(
            "No --train_csv given; every column falls back to a flat class prior of %.3f. "
            "Pass --train_csv <path to train.csv> to compute real per-column class priors instead.",
            FLAT_CLASS_PRIOR,
        )
        return {column: FLAT_CLASS_PRIOR for column in RSNA_TARGET_COLUMNS}

    frame = load_rsna_labels_csv(train_csv, require_targets=False)
    priors: Dict[str, float] = {}
    for column in RSNA_TARGET_COLUMNS:
        values = frame[column].dropna()
        if values.empty:
            logger.warning(
                "%s has no labeled values for '%s'; falling back to flat prior %.3f for that column.",
                train_csv, column, FLAT_CLASS_PRIOR,
            )
            priors[column] = FLAT_CLASS_PRIOR
        else:
            priors[column] = float(values.mean())

    logger.info("Computed class priors from %s: %s", train_csv, {k: round(v, 4) for k, v in priors.items()})
    return priors


# --------------------------------------------------------------------------- #
# Backend loading
# --------------------------------------------------------------------------- #

def load_vqc_heads(device: torch.device) -> Dict[str, object]:
    """Builds the ACL and Meniscus `VQCClassifier` heads, loading each
    head's trained checkpoint when available and falling back to
    randomly-initialized weights (with a loud warning) otherwise — same
    "trained if available, else honestly-labeled demo weights" convention
    every other entry point in this project follows. Independent of which
    `--backend` extracts the quantum angles."""
    from qknee.models.pipeline import PipelineValidationError, load_vqc_weights
    from qknee.models.vqc import VQCClassifier

    checkpoints = {"acl": _config.paths.acl_checkpoint, "meniscus": _config.paths.meniscus_checkpoint}
    seeds = {"acl": 42, "meniscus": 7}

    heads: Dict[str, VQCClassifier] = {}
    for head_name, checkpoint_path in checkpoints.items():
        torch.manual_seed(seeds[head_name])
        model = VQCClassifier()
        if checkpoint_path.exists():
            try:
                load_vqc_weights(model, checkpoint_path, device=device)
                logger.info("Loaded trained %s VQC weights from %s", head_name, checkpoint_path)
            except PipelineValidationError as exc:
                logger.warning("Failed to load %s checkpoint; using random weights: %s", head_name, exc)
        else:
            logger.warning(
                "No %s checkpoint found at %s; using randomly initialized weights.", head_name, checkpoint_path,
            )
        model.to(device).eval()
        heads[head_name] = model
    return heads


def build_angles_backend(backend: str, device: torch.device):
    """Builds whichever engine will run the classical (ResNet18 -> PCA)
    stage: a native `PipelineRunner` (`--backend pytorch`, default) or a
    `HybridONNXInferenceEngine` (`--backend onnx`) loaded from the
    artifacts `scripts/export_onnx.py` produces."""
    if backend == "onnx":
        from qknee.models.pipeline import (
            DEFAULT_CIRCUIT_PARAMS_PATH,
            DEFAULT_RESNET_ONNX_PATH,
            DEFAULT_VQC_WEIGHTS_PATH,
            HybridONNXInferenceEngine,
        )

        if not DEFAULT_RESNET_ONNX_PATH.exists() or not DEFAULT_VQC_WEIGHTS_PATH.exists():
            raise SubmissionValidationError(
                "--backend onnx requires exported ONNX artifacts "
                f"({DEFAULT_RESNET_ONNX_PATH}, {DEFAULT_VQC_WEIGHTS_PATH}) — export them first via "
                "`python scripts/export_onnx.py`."
            )
        engine = HybridONNXInferenceEngine(
            resnet_onnx_path=DEFAULT_RESNET_ONNX_PATH,
            vqc_weights_path=DEFAULT_VQC_WEIGHTS_PATH,
            circuit_params_path=DEFAULT_CIRCUIT_PARAMS_PATH if DEFAULT_CIRCUIT_PARAMS_PATH.exists() else None,
        )
        logger.info("--backend onnx: using HybridONNXInferenceEngine for the classical (ResNet18+PCA) stage.")
        return engine

    from qknee.models.pipeline import PipelineRunner

    if not _config.paths.pca_artifact.exists():
        raise SubmissionValidationError(
            f"No fitted PCA artifact found at {_config.paths.pca_artifact}. Fit one first "
            "(see qknee.models.pca_reducer / scripts/train.py) before generating a submission."
        )
    runner = PipelineRunner(config=_config, device=str(device))
    logger.info("--backend pytorch: using native PyTorch/PennyLane PipelineRunner for inference.")
    return runner


# --------------------------------------------------------------------------- #
# DataLoader-driven prefetching: __getitem__ does the expensive per-study
# DICOM decode/preprocess, parallelized across num_workers worker processes.
# --------------------------------------------------------------------------- #

class RSNAInferenceDataset(Dataset):
    """Wraps a list of `RSNAStudyRecord`s so a `DataLoader` can prefetch
    (across `num_workers` worker processes) the actual DICOM decode +
    ImageNet-normalize preprocessing for every available plane of each
    study — the CPU-bound disk-I/O step this script parallelizes off the
    main process, which is left free to run model inference."""

    def __init__(self, records: Sequence[RSNAStudyRecord]) -> None:
        self.records = list(records)
        self._ingestion: Optional[DataIngestion] = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[index]
        if self._ingestion is None:
            # Built lazily so each DataLoader worker process constructs its
            # own instance after forking/spawning, rather than sharing one
            # built in the main process across the fork boundary.
            self._ingestion = DataIngestion(train=False)

        planes: Dict[str, torch.Tensor] = {}
        for plane_name, plane_dir in record.plane_series_dirs.items():
            try:
                planes[plane_name] = self._ingestion.preprocess(plane_dir).squeeze(0)  # (S, 3, 224, 224)
            except IngestionError as exc:
                logger.warning(
                    "Study %s, plane %s: failed to load/preprocess series at %s: %s",
                    record.study_instance_uid, plane_name, plane_dir, exc,
                )
        return {"study_instance_uid": record.study_instance_uid, "planes": planes}


def collate_studies(batch: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Identity collate: each study carries a different number of slices
    per plane, so studies can't be stacked into one tensor — `DataLoader`
    still parallelizes the expensive per-study preprocessing across
    `num_workers`, it just hands back a plain list of per-study dicts."""
    return batch


# --------------------------------------------------------------------------- #
# Per-study, per-plane inference + aggregation
# --------------------------------------------------------------------------- #

def score_plane_tensor(angles_backend, plane_tensor: torch.Tensor, vqc_head, device: torch.device) -> Optional[float]:
    """Runs one already-preprocessed plane's `(S, 3, 224, 224)` slice stack
    through the classical (ResNet18+PCA) stage of whichever `angles_backend`
    this run selected, then `vqc_head`'s quantum classifier. Returns `None`
    (logged) on any failure so one bad plane never aborts the whole study.
    """
    from qknee.models.pipeline import HybridONNXInferenceEngine, PipelineRunner

    try:
        if isinstance(angles_backend, PipelineRunner):
            volume_batch = plane_tensor.unsqueeze(0)  # (1, S, 3, 224, 224)
            features = angles_backend.extract_resnet_features(volume_batch)
            angles = angles_backend.reduce_to_quantum_angles(features)
            return angles_backend.classify(angles, vqc=vqc_head)

        assert isinstance(angles_backend, HybridONNXInferenceEngine)
        # HybridONNXInferenceEngine.extract_angles has no native volume
        # dimension (it runs per-slice); the exported PCA bottleneck is a
        # fixed affine map, so mean-pooling its *output* angles across
        # slices is mathematically equivalent to mean-pooling the pre-PCA
        # 512-D features first (`ResNet18FeatureExtractor.forward_volume`'s
        # "mean" mode) — same math, just computed post-hoc here instead of
        # inside the ONNX graph.
        per_slice_angles = angles_backend.extract_angles(plane_tensor.to(device))  # (S, n_qubits)
        pooled_angles = per_slice_angles.mean(axis=0, keepdims=True)  # (1, n_qubits)
        angles_tensor = torch.from_numpy(pooled_angles).float().to(device)
        with torch.no_grad():
            risk = vqc_head(angles_tensor)
        risk_value = float(risk.item())
        if not 0.0 <= risk_value <= 1.0:
            raise SubmissionValidationError(f"VQC risk score {risk_value} outside expected range [0, 1]")
        return risk_value
    except Exception as exc:  # noqa: BLE001 - one bad plane series must not abort the whole submission
        logger.warning("Inference failed for one plane series: %s", exc)
        return None


def aggregate_scores(scores: List[float], aggregation: str) -> Optional[float]:
    if not scores:
        return None
    if aggregation == "max":
        return float(max(scores))
    return float(np.mean(scores))  # "mean" (default)


def score_study_from_item(
    item: Dict[str, object],
    angles_backend,
    vqc_heads: Dict[str, object],
    aggregation: str,
    class_priors: Dict[str, float],
    device: torch.device,
) -> Dict[str, float]:
    """Produces one study's full 12-column score dict: real hybrid
    quantum-model inference for `MODEL_COVERED_COLUMNS`, that column's
    class prior for everything else (and as the fallback for a
    model-covered column when the study has zero usable plane series, or
    when every plane fails inference for that column's head)."""
    uid = item["study_instance_uid"]
    planes: Dict[str, torch.Tensor] = item["planes"]  # type: ignore[assignment]
    row: Dict[str, float] = dict(class_priors)

    if not planes:
        logger.warning("Study %s: no usable plane series; all columns fall back to class priors.", uid)
        return row

    for head_name in set(MODEL_COVERED_COLUMNS.values()):
        vqc_head = vqc_heads[head_name]
        plane_scores = [
            score for plane_tensor in planes.values()
            if (score := score_plane_tensor(angles_backend, plane_tensor, vqc_head, device)) is not None
        ]
        aggregated = aggregate_scores(plane_scores, aggregation)
        if aggregated is None:
            logger.warning(
                "Study %s: all plane series failed inference for the '%s' head; falling back to class prior.",
                uid, head_name,
            )
            continue  # row[...] already holds that column's class prior
        for column, mapped_head in MODEL_COVERED_COLUMNS.items():
            if mapped_head == head_name:
                row[column] = aggregated

    return row


# --------------------------------------------------------------------------- #
# Formatting & validation guardrails
# --------------------------------------------------------------------------- #

def sanitize_submission_values(submission: pd.DataFrame, class_priors: Dict[str, float]) -> None:
    """In-place guardrail pass over every target column: coerces to
    numeric, fills any NaN/inf with that column's class prior, and clips
    to `[0.0, 1.0]` — catches float-precision overshoot (e.g. a sigmoid
    landing at `1.0000000002`) and any series-load failure that slipped
    through, before the hard-fail `validate_submission` check below."""
    for column in RSNA_TARGET_COLUMNS:
        values = pd.to_numeric(submission[column], errors="coerce")
        non_finite = ~np.isfinite(values)
        n_bad = int(non_finite.sum())
        if n_bad:
            logger.warning(
                "Column '%s' had %d NaN/null/infinite value(s); filled with class prior %.4f.",
                column, n_bad, class_priors[column],
            )
            values = values.where(~non_finite, class_priors[column])
        submission[column] = values.clip(lower=0.0, upper=1.0)


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
    if missing or extra or len(submission) != len(expected_uids):
        raise SubmissionValidationError(
            f"StudyInstanceUID mismatch between submission.csv and the input test set "
            f"(submission has {len(submission)} rows, expected {len(expected_uids)}). "
            f"Missing: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}; "
            f"Unexpected: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}"
        )

    for column in RSNA_TARGET_COLUMNS:
        values = submission[column]
        if values.isna().any():
            raise SubmissionValidationError(f"Column '{column}' has {int(values.isna().sum())} NaN value(s) — a submission cannot contain NaN.")
        if not pd.api.types.is_numeric_dtype(values):
            raise SubmissionValidationError(f"Column '{column}' is not numeric (dtype={values.dtype}).")
        if not np.isfinite(values.to_numpy()).all():
            raise SubmissionValidationError(f"Column '{column}' has one or more infinite value(s) — a submission cannot contain +/-inf.")
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


def save_submission(submission: pd.DataFrame, output_path: Union[str, Path], compress: bool) -> Path:
    """Writes `submission` to `output_path` with size-optimized settings:
    6-decimal float formatting (probabilities need no more precision, and
    trimming float64's ~17 significant digits meaningfully shrinks the
    file) plus optional gzip compression. `compression="infer"` lets
    pandas pick gzip/bz2/zip/xz automatically from the final path's
    extension; `--compress` forces a `.gz` suffix onto `output_path` if it
    doesn't already have a compressed extension."""
    output_path = Path(output_path)
    if compress and output_path.suffix not in (".gz", ".bz2", ".zip", ".xz"):
        output_path = output_path.with_name(output_path.name + ".gz")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, float_format="%.6f", compression="infer")
    return output_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def generate_submission(
    input_csv: Path,
    images_dir: Path,
    output_csv: Path = DEFAULT_OUTPUT_PATH,
    batch_size: int = 8,
    device: Optional[str] = None,
    backend: str = "pytorch",
    aggregation: str = "mean",
    train_csv: Optional[Path] = None,
    num_workers: int = 4,
    compress: bool = False,
    limit: Optional[int] = None,
) -> Path:
    device_obj = torch.device(device or _config.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    class_priors = compute_class_priors(train_csv)
    logger.info(
        "Model-derived columns: %s | Class-prior placeholder columns: %s",
        sorted(set(MODEL_COVERED_COLUMNS.keys())), list(PLACEHOLDER_COLUMNS),
    )

    dataset = RSNAKneeDataset(input_csv, images_dir, planes=RSNA_PLANES, require_targets=False)
    records = list(dataset)
    if limit is not None:
        records = records[:limit]
        logger.info("--limit=%d: scoring only the first %d of %d studies.", limit, len(records), len(dataset))

    expected_uids = [record.study_instance_uid for record in records]
    logger.info("Loaded %d studies from %s (images root: %s)", len(records), input_csv, images_dir)

    inference_dataset = RSNAInferenceDataset(records)
    loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_studies,
    )

    angles_backend = build_angles_backend(backend, device_obj)
    vqc_heads = load_vqc_heads(device_obj)

    rows: List[Dict[str, float]] = []
    n_scored = 0
    for batch in loader:
        for item in batch:
            row: Dict[str, float] = {RSNA_UID_COLUMN: item["study_instance_uid"]}
            row.update(score_study_from_item(item, angles_backend, vqc_heads, aggregation, class_priors, device_obj))
            rows.append(row)
            n_scored += 1
            if n_scored % 10 == 0 or n_scored == len(records):
                logger.info("Scored %d/%d studies...", n_scored, len(records))

    submission = pd.DataFrame(rows, columns=[RSNA_UID_COLUMN, *RSNA_TARGET_COLUMNS])

    sanitize_submission_values(submission, class_priors)
    validate_submission(submission, expected_uids)

    output_path = save_submission(submission, output_csv, compress)
    logger.info("Wrote %s (%d rows)", output_path, len(submission))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_csv", type=Path, required=True, help="Path to the RSNA-Knee-format test.csv (StudyInstanceUID column).")
    parser.add_argument("--images_dir", type=Path, required=True, help="Root directory: <images_dir>/<StudyInstanceUID>/<Plane>/*.dcm")
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination submission.csv path.")
    parser.add_argument("--batch_size", type=int, default=8, help="Studies per DataLoader-prefetched batch.")
    parser.add_argument("--device", type=str, default=None, help="Torch device string; defaults to config.yaml's device or CUDA-if-available.")
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch", help="Classical (ResNet18+PCA) stage: native PyTorch/PennyLane (default), or the exported HybridONNXInferenceEngine.")
    parser.add_argument("--aggregation", choices=["mean", "max"], default="mean", help="Cross-plane score aggregation per study.")
    parser.add_argument("--train_csv", type=Path, default=None, help="Optional train.csv to compute real per-column class priors from; falls back to a flat, clearly-logged prior otherwise.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes for per-study DICOM prefetching.")
    parser.add_argument("--compress", action="store_true", help="Gzip-compress the output (appends .gz to --output_csv if not already a compressed extension).")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N studies (smoke-testing).")
    args = parser.parse_args()

    setup_logging()

    generate_submission(
        input_csv=args.input_csv,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        device=args.device,
        backend=args.backend,
        aggregation=args.aggregation,
        train_csv=args.train_csv,
        num_workers=args.num_workers,
        compress=args.compress,
        limit=args.limit,
    )


if __name__ == "__main__":
    try:
        main()
    except SubmissionValidationError as exc:
        logger.error("Submission generation aborted: %s", exc)
        sys.exit(1)
