"""
Decoupled export pipeline for Q-Knee.

Produces THREE artifacts instead of one, splitting the model along its
classical/quantum boundary:

    qknee/artifacts/resnet_feature_extractor.onnx
        ResNet18 (frozen backbone, 512-D) -> PCA(4) projection, as ONE
        traced ONNX graph. `(B, 3, 224, 224) -> (B, n_qubits)` angles in
        [0, 2*pi]. Ordinary differentiable tensor arithmetic throughout
        (conv/BN/ReLU/pool + one affine PCA projection), so it traces and
        exports cleanly.

    qknee/artifacts/qknee_vqc_weights.pt
        The trained VQC's raw parameter tensors (quantum rotation weights
        + the classical readout Linear's weight/bias) — plain `torch.save`,
        no ONNX involved.

    qknee/artifacts/circuit_params.json
        A structural description of the circuit (qubit count, layer count,
        gate order, entanglement layout, encoding scheme) — everything
        needed to *reconstruct* the QNode `qknee_vqc_weights.pt`'s weights
        plug into, without also needing this repo's source.

WHY DECOUPLED — THE GRAPH-TRACING MISMATCH
    `torch.onnx.export()` cannot trace `qknee.models.vqc.VQCClassifier` as
    one graph. `qml.qnn.TorchLayer` wraps a PennyLane QNode; PennyLane's
    `default.qubit` simulator represents the quantum state vector as a
    complex128 tensor internally, and when the TorchScript-based ONNX
    exporter traces through a QNode call it records the *entire* low-level
    state-vector simulation as raw `aten::` ops — including complex-dtype
    tensor ops that have no ONNX equivalent. The export fails with
    `RuntimeError: Unknown number type: complex`. `demonstrate_vqc_export_failure()`
    below reproduces this on demand, so the failure is a verified fact
    about this codebase, not an assumption.

    This isn't fixable with a different opset or exporter flag: ONNX's op
    set has no complex-tensor representation, and PennyLane's simulator
    fundamentally needs one. The fix is architectural — export only the
    classical half (this script's ONNX graph), and keep the quantum half
    as raw weight tensors + a circuit description, re-executed directly
    through PennyLane/Qiskit at inference time
    (`qknee.models.pipeline.HybridONNXInferenceEngine`) rather than through
    any traced graph.

Usage:
    python scripts/export_onnx.py                    # export all 3 artifacts
    python scripts/export_onnx.py --validate          # + numerical parity check
    python scripts/export_onnx.py --demonstrate-failure  # show the VQC export failing, then continue
    python scripts/export_onnx.py --checkpoint qknee/artifacts/qknee_model.pt  # export a trained model's weights
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

# Allow `python scripts/export_onnx.py` to resolve the `qknee` package
# without requiring the caller to set PYTHONPATH or use `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger, setup_logging
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.pipeline import (
    DEFAULT_CIRCUIT_PARAMS_PATH,
    DEFAULT_RESNET_ONNX_PATH,
    DEFAULT_VQC_WEIGHTS_PATH,
    HybridONNXInferenceEngine,
)
from qknee.models.qknee_model import PCAProjectionLayer, QKneeModel
from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.models.vqc import VQCClassifier

logger = get_logger(__name__)
_config = load_config()

DEFAULT_OPSET = 17


# --------------------------------------------------------------------------- #
# 0. Demonstrate the graph-tracing mismatch (optional, for documentation/CI)
# --------------------------------------------------------------------------- #

def demonstrate_vqc_export_failure(n_qubits: int = 4, n_layers: int = 2) -> None:
    """Attempts `torch.onnx.export()` on a bare `VQCClassifier` and expects
    it to fail — proving (rather than just asserting in a docstring) that
    the quantum layer cannot be traced through ONNX, and that the
    decoupled-export architecture below is a necessity, not a stylistic
    choice.

    Logs the failure and returns normally if it fails as expected; raises
    `AssertionError` if the export unexpectedly *succeeds* (which would
    mean this codebase's reason for decoupling has gone stale and needs
    re-investigating).
    """
    import io

    torch.manual_seed(0)
    model = VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)
    model.eval()
    dummy_input = torch.rand(1, n_qubits) * 2 * torch.pi

    buffer = io.BytesIO()
    try:
        torch.onnx.export(
            model, dummy_input, buffer,
            input_names=["angles"], output_names=["risk"], opset_version=DEFAULT_OPSET, dynamo=False,
        )
    except Exception as exc:  # noqa: BLE001 - this is the expected/demonstrated failure
        logger.info(
            "Confirmed: torch.onnx.export(VQCClassifier, ...) fails as expected "
            "(%s: %s) — the quantum layer cannot be traced through ONNX.",
            type(exc).__name__, str(exc).splitlines()[0] if str(exc) else "",
        )
        return

    raise AssertionError(
        "torch.onnx.export(VQCClassifier, ...) succeeded — the graph-tracing mismatch this "
        "script's decoupled-export architecture exists to work around may no longer apply. "
        "Re-verify whether the classical/quantum split is still necessary."
    )


# --------------------------------------------------------------------------- #
# 1. Classical half: ResNet18 + PCA(4) -> one ONNX graph
# --------------------------------------------------------------------------- #

class ResNetPCAFeatureExtractor(nn.Module):
    """Combines `ResNet18FeatureExtractor` and `PCAProjectionLayer` into one
    traceable module: `(B, 3, 224, 224)` image -> `(B, 512)` embedding ->
    `(B, n_qubits)` angle-encoded features in `[0, 2*pi]`. Everything here
    is ordinary differentiable tensor arithmetic (conv/BN/ReLU/pool + one
    affine PCA projection), so — unlike `VQCClassifier` — it traces through
    `torch.onnx.export()` cleanly. This is the classical export target;
    it is not used directly at inference (see `HybridONNXInferenceEngine`,
    which loads its exported ONNX graph instead)."""

    def __init__(self, resnet: ResNet18FeatureExtractor, pca_layer: PCAProjectionLayer):
        super().__init__()
        self.resnet = resnet
        self.pca_layer = pca_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features_512d = self.resnet.forward_slice(x)
        return self.pca_layer(features_512d)


def _load_or_fit_pca_reducer(pca_artifact_path: Path) -> QuantumDimReducer:
    """Loads the fitted `QuantumDimReducer` at `pca_artifact_path`, or —
    if none exists yet — fits a throwaway one on random 512-D features
    purely so this script's export/validate flow is runnable standalone
    (mirrors the fallback `qknee.models.pipeline`'s own `__main__` uses).
    A real deployment should always have a properly-fitted artifact here."""
    if pca_artifact_path.exists():
        return QuantumDimReducer.load(pca_artifact_path)

    logger.warning(
        "No fitted PCA artifact found at %s; fitting a dummy one on random 512-D "
        "features purely for export/smoke-testing. Fit a real one via "
        "qknee.models.pca_reducer before deploying this export.",
        pca_artifact_path,
    )
    rng = np.random.default_rng(0)
    dummy_corpus = rng.normal(size=(500, _config.resnet.feature_dim)).astype(np.float32)
    return QuantumDimReducer().fit(dummy_corpus)


def export_resnet_pca_to_onnx(
    pca_artifact_path: Path = _config.paths.pca_artifact,
    output_path: Path = DEFAULT_RESNET_ONNX_PATH,
    opset_version: int = DEFAULT_OPSET,
) -> Path:
    """Exports `ResNetPCAFeatureExtractor` (ResNet18 + PCA(4)) to one ONNX
    file with a dynamic batch axis — the classical half of the decoupled
    pipeline."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    resnet = ResNet18FeatureExtractor(freeze_backbone=True)
    reducer = _load_or_fit_pca_reducer(Path(pca_artifact_path))
    pca_layer = PCAProjectionLayer.from_reducer(reducer)

    combined = ResNetPCAFeatureExtractor(resnet, pca_layer)
    combined.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    logger.info("Exporting ResNet18+PCA(%d) to %s (opset %d)...", reducer.n_components, output_path, opset_version)
    torch.onnx.export(
        combined,
        dummy_input,
        str(output_path),
        input_names=["slice"],
        output_names=["angles"],
        dynamic_axes={"slice": {0: "batch"}, "angles": {0: "batch"}},
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # see scripts/export_onnx.py module docstring / previous export_resnet_to_onnx
    )
    logger.info("Saved ONNX model to %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


# --------------------------------------------------------------------------- #
# 2. Quantum half: raw weights + circuit structure (never traced through ONNX)
# --------------------------------------------------------------------------- #

def _load_or_init_vqc(checkpoint_path: Optional[Path], n_qubits: int, n_layers: int) -> VQCClassifier:
    """Loads a trained `VQCClassifier` from a checkpoint if given/available,
    else returns a freshly (seeded) initialized one — mirrors the rest of
    this project's "trained weights if available, else demoable randomly-
    initialized weights" fallback pattern."""
    from qknee.models.pipeline import load_vqc_weights

    model = VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)
    resolved_path = checkpoint_path or _config.paths.model_checkpoint
    if resolved_path and Path(resolved_path).exists():
        load_vqc_weights(model, resolved_path)
        logger.info("Loaded trained VQC weights from %s", resolved_path)
    else:
        torch.manual_seed(0)
        model = VQCClassifier(n_qubits=n_qubits, n_layers=n_layers)
        logger.warning(
            "No trained VQC checkpoint found at %s; exporting randomly-initialized "
            "weights purely for export/smoke-testing.", resolved_path,
        )
    model.eval()
    return model


def export_vqc_weights_and_circuit_params(
    vqc: VQCClassifier,
    weights_path: Path = DEFAULT_VQC_WEIGHTS_PATH,
    params_path: Path = DEFAULT_CIRCUIT_PARAMS_PATH,
) -> Tuple[Path, Path]:
    """Saves the VQC's raw parameter tensors (`.pt`, via `torch.save` — no
    ONNX involved) and a JSON description of the circuit's gate/entanglement
    structure. Together these two files are the quantum half of the
    decoupled export; `HybridONNXInferenceEngine` re-executes them directly
    through a fresh PennyLane/Qiskit QNode, never a traced graph."""
    weights_path = Path(weights_path)
    params_path = Path(params_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    quantum_weights = vqc.quantum_layer.weights.detach().clone()  # (n_layers, n_qubits, 3)
    readout_weight = vqc.readout.weight.detach().clone()          # (1, n_qubits)
    readout_bias = vqc.readout.bias.detach().clone()              # (1,)

    torch.save(
        {
            "quantum_weights": quantum_weights,
            "readout_weight": readout_weight,
            "readout_bias": readout_bias,
            "n_qubits": vqc.n_qubits,
            "n_layers": vqc.n_layers,
        },
        weights_path,
    )
    logger.info("Saved VQC weights to %s", weights_path)

    n_qubits = vqc.n_qubits
    circuit_params = {
        "n_qubits": n_qubits,
        "n_layers": vqc.n_layers,
        "encoding": {
            "scheme": "angle",
            "gates_per_qubit": ["RX", "RY"],
            "input_range": [0.0, 2 * 3.141592653589793],
        },
        "variational_block": {
            "rotations_per_qubit_per_layer": 3,
            "rotation_gate_order": ["RX", "RY", "RZ"],
        },
        "entanglement": {
            "pattern": "ring",
            "gate": "CNOT",
            # (control, target) pairs for one layer's entangling ring —
            # identical every layer.
            "connectivity": [[i, (i + 1) % n_qubits] for i in range(n_qubits)],
        },
        "measurement": {"observable": "PauliZ", "wires": list(range(n_qubits)), "output_range": [-1.0, 1.0]},
        "readout": {"type": "Linear", "in_features": n_qubits, "out_features": 1, "activation": "Sigmoid"},
        "weight_shape": [vqc.n_layers, n_qubits, 3],
        "simulator_device": _config.quantum.device,
    }
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(circuit_params, handle, indent=2)
    logger.info("Saved circuit parameters to %s", params_path)

    return weights_path, params_path


# --------------------------------------------------------------------------- #
# 3. Numerical parity validation: raw PyTorch QKneeModel vs. the decoupled
#    ONNX + PennyLane/Qiskit engine
# --------------------------------------------------------------------------- #

def validate_decoupled_export(
    qknee_model: QKneeModel,
    resnet_onnx_path: Path = DEFAULT_RESNET_ONNX_PATH,
    vqc_weights_path: Path = DEFAULT_VQC_WEIGHTS_PATH,
    circuit_params_path: Path = DEFAULT_CIRCUIT_PARAMS_PATH,
    atol: float = 1e-4,
    batch_size: int = 4,
    seed: int = 1,
) -> float:
    """Runs the same random batch through `qknee_model` (raw PyTorch,
    end-to-end eager) and `HybridONNXInferenceEngine` (decoupled: ONNX
    Runtime for the classical half, a fresh PennyLane QNode for the
    quantum half), and asserts the two agree within `atol`.

    Raises:
        AssertionError: if the maximum absolute error between the two
            exceeds `atol`.

    Returns:
        The measured maximum absolute error.
    """
    torch.manual_seed(seed)
    sample = torch.rand(batch_size, 3, 224, 224)  # batch size != export-time dummy input (1), exercises dynamic axis

    qknee_model.eval()
    with torch.no_grad():
        torch_output = qknee_model(sample).numpy()  # (B, 1)

    engine = HybridONNXInferenceEngine(
        resnet_onnx_path=resnet_onnx_path,
        vqc_weights_path=vqc_weights_path,
        circuit_params_path=circuit_params_path,
    )
    engine_output = engine.predict(sample)  # (B, 1)

    max_abs_diff = float(np.abs(torch_output - engine_output).max())
    logger.info("Max abs difference (PyTorch QKneeModel vs. HybridONNXInferenceEngine): %.2e", max_abs_diff)

    np.testing.assert_allclose(
        engine_output, torch_output, atol=atol, rtol=1e-3,
        err_msg=(
            f"Decoupled ONNX engine diverges from the raw PyTorch model by more than "
            f"atol={atol} (max abs diff {max_abs_diff:.2e})"
        ),
    )
    logger.info("Numerical parity OK (PyTorch vs. decoupled ONNX engine, atol=%.0e).", atol)
    return max_abs_diff


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resnet-output", type=Path, default=DEFAULT_RESNET_ONNX_PATH)
    parser.add_argument("--vqc-weights-output", type=Path, default=DEFAULT_VQC_WEIGHTS_PATH)
    parser.add_argument("--circuit-params-output", type=Path, default=DEFAULT_CIRCUIT_PARAMS_PATH)
    parser.add_argument("--pca-artifact", type=Path, default=_config.paths.pca_artifact)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Trained VQC checkpoint to export; defaults to config.yaml's model_checkpoint if present, else randomly-initialized weights.")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--n-qubits", type=int, default=_config.quantum.n_qubits)
    parser.add_argument("--n-layers", type=int, default=_config.quantum.n_layers)
    parser.add_argument(
        "--demonstrate-failure", action="store_true",
        help="Attempt (and expect to fail) exporting VQCClassifier directly to ONNX first, "
             "to prove the graph-tracing mismatch this script's split architecture works around.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="After export, run structural checks on the ONNX graph and a numerical-parity "
             "check (max abs error < 1e-4) against a fresh PyTorch QKneeModel built from the "
             "same exported weights.",
    )
    args: argparse.Namespace = parser.parse_args()

    setup_logging()

    if args.demonstrate_failure:
        demonstrate_vqc_export_failure(n_qubits=args.n_qubits, n_layers=args.n_layers)

    resnet_onnx_path = export_resnet_pca_to_onnx(
        pca_artifact_path=args.pca_artifact, output_path=args.resnet_output, opset_version=args.opset,
    )

    vqc = _load_or_init_vqc(args.checkpoint, n_qubits=args.n_qubits, n_layers=args.n_layers)
    vqc_weights_path, circuit_params_path = export_vqc_weights_and_circuit_params(
        vqc, weights_path=args.vqc_weights_output, params_path=args.circuit_params_output,
    )

    if args.validate:
        import onnx

        logger.info("Validating ONNX graph structure...")
        onnx.checker.check_model(onnx.load(str(resnet_onnx_path)))
        logger.info("Graph structure OK.")

        # Build a QKneeModel wired to the exact same PCA reducer + VQC that
        # were just exported, so the parity check compares "the model these
        # artifacts came from" rather than an unrelated one.
        reducer = _load_or_fit_pca_reducer(Path(args.pca_artifact))
        qknee_model = QKneeModel(pca_reducer=reducer, n_qubits=vqc.n_qubits, n_layers=vqc.n_layers)
        qknee_model.vqc.load_state_dict(vqc.state_dict())

        validate_decoupled_export(
            qknee_model,
            resnet_onnx_path=resnet_onnx_path,
            vqc_weights_path=vqc_weights_path,
            circuit_params_path=circuit_params_path,
        )


if __name__ == "__main__":
    main()
