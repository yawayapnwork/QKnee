"""
Exports the frozen ResNet18 feature extractor
(`qknee.models.resnet_extractor.ResNet18FeatureExtractor`) to ONNX format,
for accelerated CPU/GPU inference via ONNX Runtime
(`qknee.models.resnet_extractor.ONNXFeatureExtractor`) instead of the
native eager PyTorch forward pass.

Only the per-slice path (`forward_slice`: `(B, 3, 224, 224) -> (B, 512)`)
is exported. `forward_volume`'s slice-averaging loop stays plain PyTorch/
numpy on both sides (`ONNXFeatureExtractor.forward_volume` folds slices
into the batch dim and calls the exported per-slice graph once, then
averages) — exporting the fold/unfold/mean logic itself into the ONNX
graph would add export complexity for no measurable inference-time
benefit over doing it in Python around one per-slice graph call.

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --output qknee/artifacts/resnet18.onnx --opset 17 --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/export_onnx.py` to resolve the `qknee` package
# without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from qknee.config.logging_config import get_logger, setup_logging
from qknee.models.resnet_extractor import ResNet18FeatureExtractor

logger = get_logger(__name__)

DEFAULT_OUTPUT_PATH = Path("qknee/artifacts/resnet18_extractor.onnx")
DEFAULT_OPSET = 17


def export_resnet_to_onnx(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """Exports `ResNet18FeatureExtractor.forward_slice` to a `.onnx` file
    with a dynamic batch axis, so the exported graph accepts any batch
    size at inference time — matching the PyTorch module's own
    flexibility, rather than baking in the dummy-input batch size of 1.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    extractor.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    logger.info("Exporting ResNet18FeatureExtractor.forward_slice to %s (opset %d)...", output_path, opset_version)
    torch.onnx.export(
        extractor,
        dummy_input,
        str(output_path),
        input_names=["slice"],
        output_names=["features"],
        dynamic_axes={"slice": {0: "batch"}, "features": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True,
        # The dynamo-based exporter (torch's new default) needs the
        # optional `onnxscript` package and uses `dynamic_shapes` instead
        # of `dynamic_axes`; the legacy TorchScript-based exporter (this
        # flag) needs neither and is fully sufficient for this frozen,
        # control-flow-free CNN backbone.
        dynamo=False,
    )
    logger.info("Saved ONNX model to %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


def validate_onnx_export(onnx_path: Path, atol: float = 1e-4) -> None:
    """Sanity-checks the exported graph two ways:

        1. `onnx.checker.check_model` — structural validity of the graph
           (valid opset, no dangling nodes, consistent shapes/types).
        2. Numerical parity — runs the same random batch through both the
           original PyTorch module and an onnxruntime `InferenceSession`,
           and asserts the two outputs match within `atol`.

    Raises:
        AssertionError: if the ONNX Runtime output diverges from the
            PyTorch module's output by more than `atol`.
    """
    import onnx
    import onnxruntime as ort

    logger.info("Validating ONNX graph structure...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("Graph structure OK.")

    torch.manual_seed(1)
    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    extractor.eval()
    sample = torch.randn(3, 3, 224, 224)  # batch size != the export-time dummy input, exercises the dynamic axis

    with torch.no_grad():
        torch_output = extractor.forward_slice(sample).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"slice": sample.numpy()})[0]

    max_abs_diff = float(np.abs(torch_output - onnx_output).max())
    logger.info("Max abs difference (PyTorch vs ONNX Runtime): %.2e", max_abs_diff)
    # numpy.testing.assert_allclose raises (with a detailed mismatch report)
    # instead of returning a bool, so a divergent export fails loudly here
    # rather than requiring the caller to check a return value.
    np.testing.assert_allclose(
        onnx_output, torch_output, atol=atol, rtol=1e-3,
        err_msg=f"ONNX export diverges from the PyTorch module by more than atol={atol} (max abs diff {max_abs_diff:.2e})",
    )
    logger.info("Numerical parity OK (PyTorch vs ONNX Runtime, atol=%.0e).", atol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Destination .onnx path")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    parser.add_argument(
        "--validate", action="store_true",
        help="Run structural + numerical parity checks against the PyTorch module after export",
    )
    args = parser.parse_args()

    setup_logging()
    output_path = export_resnet_to_onnx(output_path=args.output, opset_version=args.opset)

    if args.validate:
        validate_onnx_export(output_path)


if __name__ == "__main__":
    main()
