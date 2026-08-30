"""
Tests for the decoupled ONNX export pipeline
(`scripts/export_onnx.py` + `qknee.models.pipeline.HybridONNXInferenceEngine`).

Covers:
    1. `torch.onnx.export()` genuinely cannot trace `VQCClassifier` (the
       "graph-tracing mismatch" the decoupled architecture exists to work
       around) — verified, not assumed.
    2. The classical half (`ResNetPCAFeatureExtractor`) exports cleanly to
       a structurally valid ONNX graph.
    3. The quantum half's export (`qknee_vqc_weights.pt` +
       `circuit_params.json`) round-trips exactly, and the two files'
       `n_qubits`/`n_layers` are cross-checked.
    4. Numerical parity: `HybridONNXInferenceEngine.predict()` matches a
       raw PyTorch `QKneeModel`'s forward pass with max abs error < 1e-4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.pipeline import HybridONNXInferenceEngine
from qknee.models.qknee_model import QKneeModel
from qknee.models.vqc import VQCClassifier

N_QUBITS = 4
N_LAYERS = 2  # kept small (vs. config's default 3) so these tests stay fast


@pytest.fixture(scope="module")
def fitted_reducer() -> QuantumDimReducer:
    rng = np.random.default_rng(0)
    corpus = rng.normal(size=(200, 512)).astype(np.float32)
    return QuantumDimReducer().fit(corpus)


@pytest.fixture(scope="module")
def trained_vqc() -> VQCClassifier:
    """A VQC with real (if briefly-trained) weights, not just fresh
    initialization — exercises the export path against non-default
    parameter values."""
    torch.manual_seed(0)
    model = VQCClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.BCELoss()
    dummy_x = torch.rand(8, N_QUBITS) * 2 * torch.pi
    dummy_y = torch.randint(0, 2, (8, 1)).float()
    for _ in range(5):
        optimizer.zero_grad()
        loss = loss_fn(model(dummy_x), dummy_y)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


@pytest.fixture(scope="module")
def exported_artifacts(tmp_path_factory: pytest.TempPathFactory, fitted_reducer: QuantumDimReducer, trained_vqc: VQCClassifier):
    """Exports both decoupled artifacts once per module (export is not
    free — one ResNet18 forward-mode trace + a QNode weight save)."""
    from scripts.export_onnx import export_resnet_pca_to_onnx, export_vqc_weights_and_circuit_params

    artifact_dir = tmp_path_factory.mktemp("decoupled_export")
    pca_artifact_path = artifact_dir / "pca_scaler.pkl"
    fitted_reducer.save(pca_artifact_path)

    resnet_onnx_path = export_resnet_pca_to_onnx(
        pca_artifact_path=pca_artifact_path,
        output_path=artifact_dir / "resnet_feature_extractor.onnx",
        opset_version=17,
    )
    vqc_weights_path, circuit_params_path = export_vqc_weights_and_circuit_params(
        trained_vqc,
        weights_path=artifact_dir / "qknee_vqc_weights.pt",
        params_path=artifact_dir / "circuit_params.json",
    )
    return {
        "resnet_onnx_path": resnet_onnx_path,
        "vqc_weights_path": vqc_weights_path,
        "circuit_params_path": circuit_params_path,
    }


# --------------------------------------------------------------------------- #
# 1. The graph-tracing mismatch is real
# --------------------------------------------------------------------------- #

class TestGraphTracingMismatch:
    def test_vqc_classifier_cannot_be_exported_to_onnx(self):
        """`torch.onnx.export()` on a bare `VQCClassifier` must fail —
        this is the entire reason the export pipeline is decoupled rather
        than exporting one end-to-end graph."""
        import io

        torch.manual_seed(0)
        model = VQCClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS)
        model.eval()
        dummy_input = torch.rand(1, N_QUBITS) * 2 * torch.pi

        with pytest.raises(Exception):
            torch.onnx.export(
                model, dummy_input, io.BytesIO(),
                input_names=["angles"], output_names=["risk"], opset_version=17, dynamo=False,
            )

    def test_demonstrate_vqc_export_failure_does_not_raise(self):
        """The script's own documented-failure helper should observe the
        expected failure and return normally (not raise) — it only raises
        if the export unexpectedly *succeeds*."""
        from scripts.export_onnx import demonstrate_vqc_export_failure

        demonstrate_vqc_export_failure(n_qubits=N_QUBITS, n_layers=N_LAYERS)


# --------------------------------------------------------------------------- #
# 2 & 3. Export artifacts are well-formed
# --------------------------------------------------------------------------- #

class TestExportArtifacts:
    def test_resnet_onnx_graph_is_structurally_valid(self, exported_artifacts):
        onnx_model = onnx.load(str(exported_artifacts["resnet_onnx_path"]))
        onnx.checker.check_model(onnx_model)

    def test_resnet_onnx_output_shape_is_n_qubits(self, exported_artifacts):
        import onnxruntime as ort

        session = ort.InferenceSession(str(exported_artifacts["resnet_onnx_path"]), providers=["CPUExecutionProvider"])
        sample = np.random.default_rng(0).normal(size=(3, 3, 224, 224)).astype(np.float32)
        output = session.run(None, {session.get_inputs()[0].name: sample})[0]
        assert output.shape == (3, N_QUBITS)
        assert np.all(output >= -1e-6) and np.all(output <= 2 * np.pi + 1e-6)

    def test_vqc_weights_round_trip_exactly(self, exported_artifacts, trained_vqc):
        loaded = torch.load(exported_artifacts["vqc_weights_path"], map_location="cpu")
        assert loaded["n_qubits"] == trained_vqc.n_qubits
        assert loaded["n_layers"] == trained_vqc.n_layers
        torch.testing.assert_close(loaded["quantum_weights"], trained_vqc.quantum_layer.weights.detach())
        torch.testing.assert_close(loaded["readout_weight"], trained_vqc.readout.weight.detach())
        torch.testing.assert_close(loaded["readout_bias"], trained_vqc.readout.bias.detach())

    def test_circuit_params_json_matches_weights(self, exported_artifacts):
        import json

        params = json.loads(Path(exported_artifacts["circuit_params_path"]).read_text(encoding="utf-8"))
        assert params["n_qubits"] == N_QUBITS
        assert params["n_layers"] == N_LAYERS
        assert params["entanglement"]["pattern"] == "ring"
        assert len(params["entanglement"]["connectivity"]) == N_QUBITS
        assert params["weight_shape"] == [N_LAYERS, N_QUBITS, 3]

    def test_engine_rejects_mismatched_circuit_params(self, exported_artifacts, tmp_path):
        """A `circuit_params.json` describing a different `n_layers` than
        the paired `.pt` weights must fail loudly at construction, not
        silently produce wrong predictions."""
        import json

        from qknee.models.pipeline import PipelineValidationError

        bad_params_path = tmp_path / "mismatched_circuit_params.json"
        params = json.loads(Path(exported_artifacts["circuit_params_path"]).read_text(encoding="utf-8"))
        params["n_layers"] = params["n_layers"] + 1
        bad_params_path.write_text(json.dumps(params), encoding="utf-8")

        with pytest.raises(PipelineValidationError):
            HybridONNXInferenceEngine(
                resnet_onnx_path=exported_artifacts["resnet_onnx_path"],
                vqc_weights_path=exported_artifacts["vqc_weights_path"],
                circuit_params_path=bad_params_path,
            )


# --------------------------------------------------------------------------- #
# 4. Numerical parity: raw PyTorch vs. decoupled ONNX engine
# --------------------------------------------------------------------------- #

class TestNumericalParity:
    ATOL = 1e-4

    def test_max_abs_error_under_1e_minus_4(self, exported_artifacts, fitted_reducer, trained_vqc):
        qknee_model = QKneeModel(pca_reducer=fitted_reducer, n_qubits=N_QUBITS, n_layers=N_LAYERS)
        qknee_model.vqc.load_state_dict(trained_vqc.state_dict())
        qknee_model.eval()

        torch.manual_seed(2)
        sample = torch.rand(4, 3, 224, 224)

        with torch.no_grad():
            torch_output = qknee_model(sample).numpy()

        engine = HybridONNXInferenceEngine(
            resnet_onnx_path=exported_artifacts["resnet_onnx_path"],
            vqc_weights_path=exported_artifacts["vqc_weights_path"],
            circuit_params_path=exported_artifacts["circuit_params_path"],
        )
        engine_output = engine.predict(sample)

        max_abs_error = float(np.abs(torch_output - engine_output).max())
        assert max_abs_error < self.ATOL, (
            f"Decoupled ONNX engine diverges from raw PyTorch by {max_abs_error:.2e}, "
            f"expected < {self.ATOL:.0e}"
        )

    def test_engine_output_shape_and_range(self, exported_artifacts):
        engine = HybridONNXInferenceEngine(
            resnet_onnx_path=exported_artifacts["resnet_onnx_path"],
            vqc_weights_path=exported_artifacts["vqc_weights_path"],
            circuit_params_path=exported_artifacts["circuit_params_path"],
        )
        sample = torch.rand(5, 3, 224, 224)
        output = engine.predict(sample)
        assert output.shape == (5, 1)
        assert np.all(output >= 0.0) and np.all(output <= 1.0)

    def test_engine_is_deterministic(self, exported_artifacts):
        """Same input, same weights -> identical output across two engine
        instances (no hidden randomness in the quantum-circuit loop)."""
        engine_a = HybridONNXInferenceEngine(
            resnet_onnx_path=exported_artifacts["resnet_onnx_path"],
            vqc_weights_path=exported_artifacts["vqc_weights_path"],
            circuit_params_path=exported_artifacts["circuit_params_path"],
        )
        engine_b = HybridONNXInferenceEngine(
            resnet_onnx_path=exported_artifacts["resnet_onnx_path"],
            vqc_weights_path=exported_artifacts["vqc_weights_path"],
            circuit_params_path=exported_artifacts["circuit_params_path"],
        )
        torch.manual_seed(3)
        sample = torch.rand(2, 3, 224, 224)
        np.testing.assert_array_equal(engine_a.predict(sample), engine_b.predict(sample))
