"""
Builds real, once-computed demo cases from actual RSNA Knee DICOM studies —
replaces `frontend/lib/mock-data.ts`'s hand-authored `PRESET_CASES` (fabricated
risk scores and qubit expectation values) with genuine `PipelineRunner`
output (real ResNet18 -> PCA -> VQC -> Grad-CAM inference) computed once
offline and served by `GET /api/cases` / `GET /api/cases/{case_id}`.

Why offline, not live-on-request: the real DICOM data (`train_series/`, ~2.3GB)
is `.gitignore`d and only exists on a machine that downloaded it — it is NOT
part of the deployed Render backend. Running inference on it now and shipping
only the small (image + JSON) *output* is what makes "browse real cases"
possible on the actual judged deployment, not just localhost.

Output layout (git-tracked, small — no raw DICOM ships):
    qknee/artifacts/real_demo_cases/
        index.json              - [{case_id, label, diagnosis, risk_score}, ...]
        <case_id>.json           - full PredictionResponse-shaped payload
                                   (risk_score, diagnosis, base_image,
                                   gradcam_overlay, quantum_expectations, ...)

Each case's fields are named to match `extras/api/server.py`'s
`PredictionResponse` exactly, so the frontend's existing
`quantumTelemetryFromPrediction`/`volumeViewFromPrediction`/
`provenanceFromPrediction` (built for real `/predict` responses) work
unchanged on a case fetched from `GET /api/cases/{case_id}` — no separate
"preset" data shape, no separate fabricated-data code path.

Usage:
    python scripts/build_real_demo_cases.py --n-cases 40
    python scripts/build_real_demo_cases.py --n-cases 40 --series-dir train_series
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.ingestion import DataIngestion
from qknee.models.pipeline import PipelineRunner
from qknee.xai.gradcam import colorize_heatmap_rgba, overlay_heatmap

logger = get_logger(__name__)

OUTPUT_DIR = REPO_ROOT / "qknee" / "artifacts" / "real_demo_cases"
TEAR_RISK_THRESHOLD = 0.5


def _build_diagnosis_reason(
    risk_score: float,
    threshold: float,
    quantum_expectations: Optional[List[float]],
    gradcam_heatmap: Optional[np.ndarray],
) -> str:
    """Identical to `extras.api.server._build_diagnosis_reason` — duplicated
    here (rather than imported) so this offline script never pays that
    module's FastAPI/auth import chain just to build one sentence. See that
    function's docstring for the full rationale."""
    margin_pct = abs(risk_score - threshold) * 100
    side = "above" if risk_score >= threshold else "below"
    sentence = (
        f"Risk score {risk_score * 100:.1f}% is {margin_pct:.1f} percentage points {side} "
        f"the {threshold * 100:.0f}% detection threshold."
    )

    if quantum_expectations:
        dominant_idx = max(range(len(quantum_expectations)), key=lambda i: abs(quantum_expectations[i]))
        sentence += (
            f" Qubit {dominant_idx} contributed the strongest signal "
            f"(⟨Z⟩ = {quantum_expectations[dominant_idx]:+.3f})."
        )

    if gradcam_heatmap is not None and float(gradcam_heatmap.max()) > 0:
        row_idx, col_idx = np.unravel_index(np.argmax(gradcam_heatmap), gradcam_heatmap.shape)
        height, width = gradcam_heatmap.shape
        vertical = "upper" if row_idx < height / 3 else ("lower" if row_idx > 2 * height / 3 else "central")
        horizontal = "left" if col_idx < width / 3 else ("right" if col_idx > 2 * width / 3 else "center")
        region = "center" if vertical == "central" and horizontal == "center" else f"{vertical}-{horizontal}"
        sentence += f" Grad-CAM attention peaks in the {region} of the analyzed slice."

    return sentence


def _normalize_uint8(slice_2d: np.ndarray) -> np.ndarray:
    """Identical to `extras.api.server.QKneeBackend._normalize_uint8` —
    duplicated here (rather than imported) so this offline script never
    imports `extras.api.server`, which resolves a JWT signing secret at
    module-import time and would otherwise fail without one configured."""
    slice_2d = slice_2d.astype(np.float32)
    min_val, max_val = float(slice_2d.min()), float(slice_2d.max())
    if max_val > min_val:
        slice_2d = (slice_2d - min_val) / (max_val - min_val)
    else:
        slice_2d = np.zeros_like(slice_2d)
    return (slice_2d * 255).astype(np.uint8)


def _encode_png_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image as PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


# Real cases are shipped in the git repo (small footprint needed to deploy
# "a lot" of them, not a handful) -- unlike a live `/predict` response
# (one case, ephemeral, full native resolution is fine), every image here
# is downsized to this max dimension before PNG encoding. Still real,
# unmodified pixel content, just lower-resolution -- standard practice for
# a thumbnail/demo-browsing view, not a quality compromise on the actual
# live-upload path (which is untouched by this script).
_MAX_DEMO_IMAGE_DIM = 320


def _resize_max_dim(image: np.ndarray, max_dim: int = _MAX_DEMO_IMAGE_DIM) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _pick_series_dir(study_dir: Path) -> Optional[Path]:
    """Picks the series subdirectory with the most `.dcm` files (the
    fullest acquisition) for one study -- a real RSNA Knee study directory
    holds one subdirectory per series (different plane/sequence), not one
    flat directory of slices."""
    series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]
    best: Optional[Path] = None
    best_count = 0
    for series_dir in series_dirs:
        count = sum(1 for _ in series_dir.glob("*.dcm"))
        if count > best_count:
            best, best_count = series_dir, count
    return best


def build_one_case(study_uid: str, series_dir: Path, runner: PipelineRunner, ingestion: DataIngestion) -> dict:
    """Runs one real DICOM series through the exact pipeline
    `extras.api.server.QKneeBackend._predict_live` runs for a live upload,
    and returns a dict with the same field names as `PredictionResponse`."""
    volume = ingestion.load_dicom_series(series_dir)  # (D, H, W), real calibrated pixel data

    axial_slices = [np.array(p) for p in DataIngestion._array_to_pil_slices(volume)]
    primary_slice_index = len(axial_slices) // 2
    base_image_uint8 = axial_slices[primary_slice_index]

    central_slice_raw = volume[primary_slice_index] if volume.ndim == 3 else volume
    display_slice = _normalize_uint8(central_slice_raw)

    t0 = time.perf_counter()
    result = runner.run(display_slice)
    latency_ms = (time.perf_counter() - t0) * 1000

    overlay_rgba = colorize_heatmap_rgba(result.gradcam_heatmap, base_image_uint8.shape[:2])
    legacy_overlay = overlay_heatmap(result.gradcam_heatmap, display_slice)
    # Grad-CAM's ReLU(sum_k(alpha_k * A_k)) can legitimately zero out every
    # pixel for a given input -- see extras/api/server.py's
    # PredictionResponse.gradcam_degenerate docstring for why this is
    # surfaced explicitly rather than shipped as a blank-looking overlay.
    gradcam_degenerate = bool(np.all(result.gradcam_heatmap == 0))

    risk_score = float(result.risk_score)
    diagnosis = "Tear Detected" if risk_score >= TEAR_RISK_THRESHOLD else "Normal"
    quantum_expectations = (
        result.pauli_z_expectations.tolist() if result.pauli_z_expectations is not None else None
    )

    return {
        "case_id": study_uid,
        "label": f"Study {study_uid[-8:]}",
        "risk_score": risk_score,
        "diagnosis": diagnosis,
        "reason": _build_diagnosis_reason(
            risk_score, TEAR_RISK_THRESHOLD, quantum_expectations,
            None if gradcam_degenerate else result.gradcam_heatmap,
        ),
        "gradcam_heatmap": _encode_png_base64(_resize_max_dim(legacy_overlay)),
        # "cache-fallback/<id>" -> qknee.observability.provenance.classify
        # maps this to provenance="precomputed_demo" -- real model, real
        # quantum circuit, computed offline and replayed, never "live".
        "backend": f"cache-fallback/{study_uid}",
        "latency_ms": latency_ms,
        "quantum_expectations": quantum_expectations,
        "n_qubits": len(quantum_expectations) if quantum_expectations is not None else None,
        "quantum_backend": runner.config.quantum.device if quantum_expectations is not None else None,
        "base_image": _encode_png_base64(_resize_max_dim(cv2.cvtColor(base_image_uint8, cv2.COLOR_GRAY2BGR))),
        "gradcam_overlay": _encode_png_base64(_resize_max_dim(overlay_rgba)),
        "gradcam_plane": "axial",
        "gradcam_slice_index": primary_slice_index,
        "gradcam_degenerate": gradcam_degenerate,
        "planes": {
            "axial": {
                "available": True,
                "num_slices": len(axial_slices),
                "slices": [
                    _encode_png_base64(_resize_max_dim(cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)))
                    for s in axial_slices
                ],
            },
            "coronal": {"available": False, "num_slices": 0, "slices": []},
            "sagittal": {"available": False, "num_slices": 0, "slices": []},
        },
        "primary_plane": "axial",
        "primary_slice_index": primary_slice_index,
        "model_checkpoint_loaded": bool(runner.vqc_checkpoint_loaded),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series-dir", type=str, default=str(REPO_ROOT / "train_series"))
    parser.add_argument("--n-cases", type=int, default=40)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()

    series_root = Path(args.series_dir)
    if not series_root.is_dir():
        raise SystemExit(f"{series_root} not found -- this script needs the real RSNA Knee train_series/ directory.")

    study_dirs = sorted(d for d in series_root.iterdir() if d.is_dir())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(study_dirs)  # type: ignore[arg-type]
    study_dirs = study_dirs[: args.n_cases]

    logger.info("Building %d real demo cases from %s", len(study_dirs), series_root)

    runner = PipelineRunner()
    ingestion = DataIngestion(train=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index: List[dict] = []
    for i, study_dir in enumerate(study_dirs):
        study_uid = study_dir.name
        series_dir = _pick_series_dir(study_dir)
        if series_dir is None:
            logger.warning("[%d/%d] %s: no series with .dcm files found, skipping", i + 1, len(study_dirs), study_uid)
            continue
        try:
            case = build_one_case(study_uid, series_dir, runner, ingestion)
        except Exception as exc:  # noqa: BLE001 - one bad study must not abort the whole batch
            logger.warning("[%d/%d] %s: failed (%s), skipping", i + 1, len(study_dirs), study_uid, exc)
            continue

        (output_dir / f"{study_uid}.json").write_text(json.dumps(case), encoding="utf-8")
        index.append({
            "case_id": case["case_id"],
            "label": case["label"],
            "diagnosis": case["diagnosis"],
            "risk_score": case["risk_score"],
        })
        logger.info(
            "[%d/%d] %s: risk_score=%.4f diagnosis=%s (%.1fms)",
            i + 1, len(study_dirs), study_uid, case["risk_score"], case["diagnosis"], case["latency_ms"],
        )

    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    logger.info("Wrote %d real demo cases to %s", len(index), output_dir)


if __name__ == "__main__":
    main()
