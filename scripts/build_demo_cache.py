"""
Builds the "Demo Mode / Latency Fallback" precomputed cache the dashboard's
"Use Precomputed NISQ Cache" sidebar toggle reads (`qknee.ui.dashboard`).

For each of a handful of sample MRNet-style cases, this script runs the
*real* inference path once (live pipeline if a trained backend is
available, else the same deterministic mock the dashboard itself falls
back to) and saves the result to disk:

    qknee/artifacts/demo_cache/
        index.json                    - case metadata + scores + latency
        case_0000_slice.png           - normalized grayscale slice
        case_0000_heatmap.npy         - raw (h, w) float32 Grad-CAM heatmap

The dashboard then replays these straight from disk — no model, no
PennyLane QNode, no ResNet forward pass — so a judge toggling "Use
Precomputed NISQ Cache" during a live demo gets an instant response
regardless of cold-start latency, container resource contention, etc.

Dataset: by default, generates a deterministic mock MRNet-shaped dataset
(`qknee.data.dataset.generate_mock_mrnet_dataset`) so this runs standalone
without the real Stanford MRNet download. Pass `--data-root` to instead
pull representative slices from a real MRNet-shaped directory.

Usage:
    python scripts/build_demo_cache.py
    python scripts/build_demo_cache.py --n-cases 10 --plane axial
    python scripts/build_demo_cache.py --data-root /path/to/real/mrnet --plane sagittal
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# Allow `python scripts/build_demo_cache.py` to resolve the `qknee` package
# without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from qknee.config.logging_config import get_logger, setup_logging
from qknee.data.dataset import generate_mock_mrnet_dataset
from qknee.data.ingestion import MultiPlaneViewSelector
from qknee.ui.dashboard import (
    DEMO_CACHE_DIR,
    InferenceResult,
    load_backend,
    normalize_for_display,
    run_live_inference,
    run_mock_inference,
)

logger = get_logger(__name__)


def _resolve_dataset_root(data_root: str | None, plane: str, n_cases: int, condition: str, split: str, seed: int) -> Path:
    if data_root is not None:
        logger.info("Using real MRNet-shaped dataset root: %s", data_root)
        return Path(data_root)

    mock_dir = Path(tempfile.mkdtemp(prefix="qknee_demo_cache_mock_"))
    case_ids = [f"{i:04d}" for i in range(n_cases)]
    root = generate_mock_mrnet_dataset(
        mock_dir, case_ids=case_ids, planes=(plane,), condition=condition,
        split=split, num_slices=10, size=224, seed=seed,
    )
    logger.info("No --data-root given; generated a mock MRNet dataset at %s (%d cases).", root, n_cases)
    return root


def build_demo_cache(
    n_cases: int = 8,
    plane: str = "axial",
    condition: str = "acl",
    split: str = "train",
    data_root: str | None = None,
    output_dir: Path = DEMO_CACHE_DIR,
    seed: int = 0,
) -> Path:
    """Runs real (or mock) inference once per case and writes the demo
    cache to `output_dir`. Returns the path to the written `index.json`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = _resolve_dataset_root(data_root, plane, n_cases, condition, split, seed)

    label_csv_path = root / f"{split}-{condition}.csv"
    if not label_csv_path.exists():
        raise FileNotFoundError(
            f"No label CSV found at {label_csv_path}. Expected an MRNet-shaped root "
            "(see qknee.data.dataset.generate_mock_mrnet_dataset for the exact layout)."
        )
    case_rows = [
        line.split(",") for line in label_csv_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    pipeline, acl_model, meniscus_model = load_backend()
    backend_ready = pipeline is not None
    logger.info("Backend for cache generation: %s", "live PipelineRunner" if backend_ready else "mock")

    cases = []
    for case_id, label_str in case_rows:
        npy_path = root / split / plane / f"{case_id}.npy"
        if not npy_path.exists():
            logger.warning("Skipping case '%s': missing volume %s", case_id, npy_path)
            continue

        volume = np.load(npy_path)
        raw_slice = MultiPlaneViewSelector(volume).get_slice(plane)  # anatomical midpoint

        result: InferenceResult = (
            run_live_inference(raw_slice, pipeline, acl_model, meniscus_model)
            if backend_ready else
            run_mock_inference(raw_slice)
        )

        display_slice = normalize_for_display(raw_slice)
        slice_filename = f"case_{case_id}_slice.png"
        cv2.imwrite(str(output_dir / slice_filename), display_slice)

        heatmap_filename = None
        if result.gradcam_heatmap is not None:
            heatmap_filename = f"case_{case_id}_heatmap.npy"
            np.save(output_dir / heatmap_filename, result.gradcam_heatmap.astype(np.float32))

        cases.append({
            "case_id": case_id,
            "plane": plane,
            "ground_truth_label": int(label_str),
            "slice_file": slice_filename,
            "heatmap_file": heatmap_filename,
            "acl_risk": result.acl_risk,
            "meniscus_risk": result.meniscus_risk,
            "resnet_latency_ms": result.resnet_latency_ms,
            "pca_latency_ms": result.pca_latency_ms,
            "quantum_latency_ms": result.quantum_latency_ms,
            "total_latency_ms": result.total_latency_ms,
            "backend": result.backend,
        })
        print(f"  cached case {case_id}: acl_risk={result.acl_risk:.3f} "
              f"total_latency={result.total_latency_ms:.1f}ms backend={result.backend}")

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps({"plane": plane, "condition": condition, "cases": cases}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote demo cache index (%d cases) to %s", len(cases), index_path)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-cases", type=int, default=8, help="Number of cases (5-10 recommended for a demo).")
    parser.add_argument(
        "--plane", choices=["axial", "coronal", "sagittal"], default="axial",
        help="'axial' (default) gives full-resolution (H, W) square slices; coronal/sagittal "
             "are thinner reformats along the mock volume's low-resolution depth axis.",
    )
    parser.add_argument("--condition", type=str, default="acl")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data-root", type=str, default=None, help="Real MRNet-shaped dataset root; omit to use a generated mock dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEMO_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging()
    print(f"Building demo cache ({args.n_cases} cases, {args.plane} plane)...")
    index_path = build_demo_cache(
        n_cases=args.n_cases, plane=args.plane, condition=args.condition, split=args.split,
        data_root=args.data_root, output_dir=args.output_dir, seed=args.seed,
    )
    print(f"\nDemo cache ready: {index_path.resolve()}")
    print("Toggle 'Use Precomputed NISQ Cache' in the dashboard sidebar to use it.")


if __name__ == "__main__":
    main()
