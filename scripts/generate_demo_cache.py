"""
Generates the Q-Knee "precomputed cache" — the PRD's Plan B mitigation for
live-demo/judging latency risk: 10 representative MRNet-style knee MRI
cases run once, offline, through the full trained hybrid pipeline, with
every intermediate result (quantum embeddings, Pauli-Z expectations, risk
scores, Grad-CAM heatmaps, and an auto-generated clinical text snippet)
serialized to disk so a live demo can replay them with zero inference
latency if the quantum simulator or a live connection misbehaves mid-demo.

Cases: 10 total, covering all three demo categories (Normal / ACL Tear /
Meniscal Tear) across both Sagittal and Coronal planes. Dataset: synthetic
MRNet-shaped volumes (`qknee.data.dataset.generate_mock_mrnet_volume`), not
the real Stanford MRNet release — this repo doesn't bundle it, and none of
the earlier scripts do either. The "category" label on each case is a
demo-only tag assigned to this script's synthetic volumes for coverage/
variety, NOT a ground-truth clinical label; the risk score and Grad-CAM
alongside it are genuine outputs of a real forward pass through the
pipeline (`qknee.models.pipeline.PipelineRunner`) — the two are independent
and shouldn't be read as "the model predicted this correctly."

Pipeline per case (`qknee.models.pipeline.PipelineRunner`):
    1. Ingest the case's representative middle slice (via
       `qknee.data.ingestion.MultiPlaneViewSelector`, plane-aware).
    2. ResNet18 -> PCA(4) -> the 4-D quantum angle embedding
       (`PipelineRunner.reduce_to_quantum_angles`).
    3. The 4-qubit VQC's quantum layer directly (bypassing
       `PipelineRunner.classify`'s single sigmoid-only return), to also
       capture the raw Pauli-Z expectation values alongside the final
       risk probability.
    4. Grad-CAM on the same middle slice (`PipelineRunner.explain`).
    5. An automated clinical text snippet
       (`qknee.xai.report_generator.generate_radiology_text_snippet`).

Outputs:
    qknee/artifacts/precomputed_cache.json   - all 10 cases' serialized results
    qknee/artifacts/heatmaps/<case_id>.png   - each case's Grad-CAM overlay PNG
        (also embedded as base64 inside precomputed_cache.json, so a
        consumer can render without touching the filesystem at all)

Usage:
    python scripts/generate_demo_cache.py
"""

from __future__ import annotations

import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Allow `python scripts/generate_demo_cache.py` to resolve the `qknee`
# package without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.dataset import generate_mock_mrnet_volume
from qknee.data.ingestion import MultiPlaneViewSelector
from qknee.xai.gradcam import overlay_heatmap
from qknee.xai.report_generator import generate_radiology_text_snippet

logger = get_logger(__name__)
_config = load_config()

DEFAULT_CACHE_PATH = Path("qknee/artifacts/precomputed_cache.json")
DEFAULT_HEATMAPS_DIR = Path("qknee/artifacts/heatmaps")

RISK_LOW_MAX = 0.33
RISK_MODERATE_MAX = 0.66

# 10 cases: covers all three demo categories, and both Sagittal/Coronal
# planes within each category where the count allows. `seed` makes each
# case's synthetic volume (and therefore its Grad-CAM/risk output, given
# fixed model weights) fully reproducible across re-runs.
DEMO_CASES: List[Dict] = [
    {"case_id": "case_0001", "category": "Normal", "plane": "sagittal", "seed": 101},
    {"case_id": "case_0002", "category": "Normal", "plane": "coronal", "seed": 102},
    {"case_id": "case_0003", "category": "Normal", "plane": "sagittal", "seed": 103},
    {"case_id": "case_0004", "category": "Normal", "plane": "coronal", "seed": 104},
    {"case_id": "case_0005", "category": "ACL Tear", "plane": "sagittal", "seed": 205},
    {"case_id": "case_0006", "category": "ACL Tear", "plane": "coronal", "seed": 206},
    {"case_id": "case_0007", "category": "ACL Tear", "plane": "sagittal", "seed": 207},
    {"case_id": "case_0008", "category": "Meniscal Tear", "plane": "coronal", "seed": 308},
    {"case_id": "case_0009", "category": "Meniscal Tear", "plane": "sagittal", "seed": 309},
    {"case_id": "case_0010", "category": "Meniscal Tear", "plane": "coronal", "seed": 310},
]

VOLUME_NUM_SLICES = 40  # deep enough that coronal/sagittal reformats aren't paper-thin
VOLUME_SIZE = 224


def _risk_tier(risk: float) -> str:
    if risk >= RISK_MODERATE_MAX:
        return "HIGH"
    if risk >= RISK_LOW_MAX:
        return "MODERATE"
    return "LOW"


def _classification_label(risk: float) -> str:
    return "TEAR LIKELY" if risk >= 0.5 else "NO TEAR INDICATED"


def _load_pipeline_runner():
    """Builds a real `PipelineRunner` — fitting a throwaway PCA reducer if
    none is on disk yet (mirrors `qknee.models.pipeline`'s own `__main__`
    smoke test), and loading a trained VQC checkpoint if one is available,
    else falling back to randomly-initialized weights (same "trained if
    available, else honestly-labeled demo weights" convention every other
    script in this project follows). Returns `(runner, backend_label)`.
    """
    from qknee.models.pca_reducer import QuantumDimReducer
    from qknee.models.pipeline import PipelineRunner

    pca_artifact_path = _config.paths.pca_artifact
    if not pca_artifact_path.exists():
        logger.warning(
            "No fitted PCA artifact found at %s; fitting a dummy one on random 512-D "
            "features purely so this script is runnable standalone.", pca_artifact_path,
        )
        pca_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        dummy_corpus = rng.normal(size=(500, _config.resnet.feature_dim)).astype(np.float32)
        QuantumDimReducer().fit(dummy_corpus).save(pca_artifact_path)

    runner = PipelineRunner(config=_config)
    checkpoint_path = _config.paths.model_checkpoint
    backend_label = "live/trained" if checkpoint_path.exists() else "live/untrained"
    return runner, backend_label


def run_full_inference(runner, raw_slice: np.ndarray) -> Dict:
    """Runs one slice through the full hybrid pipeline, capturing every
    intermediate the PRD asks for: the 4-D quantum angle embedding, the raw
    Pauli-Z expectation values (not just the final sigmoid score), the
    final risk probability, and a Grad-CAM heatmap on that same slice."""
    t0 = time.perf_counter()
    batch = runner.ingest(raw_slice)
    features = runner.extract_resnet_features(batch)
    t1 = time.perf_counter()

    quantum_angles = runner.reduce_to_quantum_angles(features)  # (1, n_qubits), in [0, 2*pi]
    t2 = time.perf_counter()

    # Call the VQC's quantum layer directly (rather than
    # PipelineRunner.classify(), which only returns the final sigmoid
    # score) so the raw Pauli-Z expectation values are captured too.
    angles_tensor = torch.from_numpy(quantum_angles).float()
    with torch.no_grad():
        pauli_z_expvals = runner.vqc.quantum_layer(angles_tensor)  # (1, n_qubits), each in [-1, 1]
        risk_logits = runner.vqc.readout(pauli_z_expvals)
        risk_tensor = runner.vqc.activation(risk_logits)
    t3 = time.perf_counter()

    risk_score = float(risk_tensor.item())

    central_slice_index = batch.shape[1] // 2
    heatmap = None
    try:
        heatmap = runner.explain(batch[:, central_slice_index])  # (h, w) in [0, 1]
    except Exception as exc:  # noqa: BLE001 - a failed heatmap shouldn't abort the whole case
        logger.warning("Grad-CAM generation failed for this case: %s", exc)
    t4 = time.perf_counter()

    return {
        "quantum_angles": quantum_angles.flatten().astype(float).tolist(),
        "pauli_z_expectations": pauli_z_expvals.detach().flatten().numpy().astype(float).tolist(),
        "risk_score": risk_score,
        "gradcam_heatmap": heatmap,
        "resnet_latency_ms": (t1 - t0) * 1000,
        "quantum_latency_ms": (t3 - t2) * 1000,
        "gradcam_latency_ms": (t4 - t3) * 1000,
        "total_latency_ms": (t4 - t0) * 1000,
    }


def build_case_record(
    case_config: Dict,
    runner,
    backend_label: str,
    heatmaps_dir: Path,
) -> Dict:
    """Generates one case's synthetic volume, runs it through the full
    pipeline, renders + saves its Grad-CAM overlay, and assembles the
    JSON-serializable case record."""
    case_id = case_config["case_id"]
    plane = case_config["plane"]

    volume = generate_mock_mrnet_volume(
        num_slices=VOLUME_NUM_SLICES, size=VOLUME_SIZE, seed=case_config["seed"],
    )
    raw_slice = MultiPlaneViewSelector(volume).get_slice(plane)  # anatomical midpoint

    result = run_full_inference(runner, raw_slice)

    display_slice = raw_slice.astype(np.float32)
    lo, hi = float(display_slice.min()), float(display_slice.max())
    display_slice = ((display_slice - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(display_slice, dtype=np.uint8)

    if result["gradcam_heatmap"] is not None:
        overlay_bgr = overlay_heatmap(result["gradcam_heatmap"], display_slice)
    else:
        overlay_bgr = cv2.cvtColor(display_slice, cv2.COLOR_GRAY2BGR)

    heatmap_filename = f"{case_id}.png"
    heatmap_path = heatmaps_dir / heatmap_filename
    cv2.imwrite(str(heatmap_path), overlay_bgr)

    success, encoded = cv2.imencode(".png", overlay_bgr)
    heatmap_base64 = base64.b64encode(encoded.tobytes()).decode("ascii") if success else None

    risk_score = result["risk_score"]
    text_snippet = generate_radiology_text_snippet(
        prediction_results={"acl_risk": risk_score},
        metadata={"patient_id": case_id, "plane": plane.capitalize()},
    )

    return {
        "case_id": case_id,
        "ground_truth_category": case_config["category"],
        "plane": plane,
        "quantum_angles": result["quantum_angles"],
        "pauli_z_expectations": result["pauli_z_expectations"],
        "risk_score": risk_score,
        "risk_tier": _risk_tier(risk_score),
        "classification_label": _classification_label(risk_score),
        "clinical_text_snippet": text_snippet,
        "resnet_latency_ms": result["resnet_latency_ms"],
        "quantum_latency_ms": result["quantum_latency_ms"],
        "gradcam_latency_ms": result["gradcam_latency_ms"],
        "total_latency_ms": result["total_latency_ms"],
        "backend": backend_label,
        "heatmap_file": f"heatmaps/{heatmap_filename}",
        "heatmap_base64": heatmap_base64,
    }


def generate_demo_cache(
    cache_path: Path = DEFAULT_CACHE_PATH,
    heatmaps_dir: Path = DEFAULT_HEATMAPS_DIR,
) -> Path:
    """Runs all 10 `DEMO_CASES` through the full pipeline and writes
    `precomputed_cache.json` + per-case heatmap PNGs. Returns the path to
    the written JSON file."""
    cache_path = Path(cache_path)
    heatmaps_dir = Path(heatmaps_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    runner, backend_label = _load_pipeline_runner()
    logger.info("Backend for cache generation: %s", backend_label)

    cases = []
    for case_config in DEMO_CASES:
        record = build_case_record(case_config, runner, backend_label, heatmaps_dir)
        cases.append(record)
        print(
            f"  {record['case_id']} [{record['ground_truth_category']:<14} / {record['plane']:<8}] "
            f"risk={record['risk_score']:.3f} ({record['risk_tier']}) "
            f"total_latency={record['total_latency_ms']:.1f}ms"
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend_label,
        "n_cases": len(cases),
        "categories": sorted({c["ground_truth_category"] for c in cases}),
        "planes": sorted({c["plane"] for c in cases}),
        "cases": cases,
    }

    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Wrote precomputed cache (%d cases) to %s", len(cases), cache_path)
    return cache_path


def main() -> None:
    setup_logging()
    print(f"Generating precomputed demo cache ({len(DEMO_CASES)} cases)...")
    cache_path = generate_demo_cache()
    print(f"\nCache ready: {cache_path.resolve()}")
    print(f"Heatmap images: {DEFAULT_HEATMAPS_DIR.resolve()}")


if __name__ == "__main__":
    main()
