"""
Parameterized tests across the two VQC ansatz "topologies":

    - qknee.models.vqc.VQCClassifier                        ("basic")
    - qknee.models.vqc_data_reuploading.DataReuploadingVQC   ("data_reuploading")

Covers:
    1. Measurement bounds: the raw Pauli-Z expectation value(s) each
       topology's quantum layer produces are mathematically guaranteed to
       lie in [-1, 1] for a valid (unitary) circuit — checked directly on
       the quantum layer's own output, not just the final sigmoid score.
    2. Unitary preservation: re-executes each topology's exact gate
       sequence with a `qml.probs()` measurement instead of `qml.expval()`
       and asserts the returned probability distribution sums to 1 — the
       standard empirical signature that every gate applied is truly
       unitary (probability-conserving), rather than trusting bounded
       expectation values alone.
    3. Full-model forward interface: every topology maps
       `(B, n_qubits) -> (B, 1)` in `[0, 1]`.
    4. No hardware/cloud quantum backend: every topology runs on
       PennyLane's local `default.qubit` state-vector simulator
       (`config.yaml`'s `quantum.device`) — asserted explicitly.
    5. Determinism: repeated forward passes with identical inputs/weights
       produce bit-identical output (no shot noise, since `default.qubit`
       runs exact state-vector simulation by default).

All tests are CPU-only and use fixed seeds throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pennylane as qml
import pytest
import torch

from qknee.config.loader import load_config
from qknee.models.vqc import VQCClassifier
from qknee.models.vqc import angle_encoding as basic_angle_encoding
from qknee.models.vqc import variational_block as basic_variational_block
from qknee.models.vqc_data_reuploading import DataReuploadingVQC
from qknee.models.vqc_data_reuploading import reuploading_encoding
from qknee.models.vqc_data_reuploading import variational_block as reupload_variational_block

_config = load_config()
N_QUBITS = _config.quantum.n_qubits
N_LAYERS = 2  # kept small (vs. config's default 3) so these tests stay fast; topology-agnostic


# --------------------------------------------------------------------------- #
# Per-topology "probs" QNodes — the exact same gate sequence each module's
# own `build_qnode` uses, but measuring `qml.probs()` (the full
# 2**n_qubits basis-state probability distribution) instead of per-qubit
# `qml.expval(PauliZ)`, so unitarity can be checked directly.
# --------------------------------------------------------------------------- #

def _probs_qnode_basic(n_qubits: int, n_layers: int):
    device = qml.device(_config.quantum.device, wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(device)
    def circuit(inputs, weights):
        basic_angle_encoding(inputs, wires)
        for layer in range(n_layers):
            basic_variational_block(weights[layer], wires)
        return qml.probs(wires=wires)

    return circuit


def _probs_qnode_data_reuploading(n_qubits: int, n_layers: int):
    device = qml.device(_config.quantum.device, wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(device)
    def circuit(inputs, enc_weights, var_weights):
        for layer in range(n_layers):
            reuploading_encoding(inputs, enc_weights[layer], wires)
            reupload_variational_block(var_weights[layer], wires)
        return qml.probs(wires=wires)

    return circuit


def _basic_weight_args(n_qubits: int, n_layers: int, rng: np.random.Generator) -> List[torch.Tensor]:
    return [torch.from_numpy(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3))).float()]


def _data_reuploading_weight_args(n_qubits: int, n_layers: int, rng: np.random.Generator) -> List[torch.Tensor]:
    enc = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(n_layers, n_qubits, 4))).float()
    var = torch.from_numpy(rng.uniform(0, 2 * np.pi, size=(n_layers, n_qubits, 3))).float()
    return [enc, var]


@dataclass(frozen=True)
class TopologySpec:
    name: str
    model_cls: Callable
    probs_qnode_builder: Callable
    weight_args_builder: Callable

    def build_model(self) -> torch.nn.Module:
        return self.model_cls(n_qubits=N_QUBITS, n_layers=N_LAYERS)

    def raw_quantum_layer(self, model: torch.nn.Module):
        """Returns the bare `qml.qnn.TorchLayer` (pre-readout, pre-sigmoid)
        for `model` — both topologies expose it as `.quantum_layer`
        directly."""
        if hasattr(model, "quantum_layer"):
            return model.quantum_layer
        return model.circuit.quantum_layer


TOPOLOGIES = [
    TopologySpec("basic", VQCClassifier, _probs_qnode_basic, _basic_weight_args),
    TopologySpec("data_reuploading", DataReuploadingVQC, _probs_qnode_data_reuploading, _data_reuploading_weight_args),
]
TOPOLOGY_IDS = [t.name for t in TOPOLOGIES]


# --------------------------------------------------------------------------- #
# 1. Measurement bounds
# --------------------------------------------------------------------------- #

class TestMeasurementBounds:
    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_raw_quantum_layer_output_in_minus1_1(self, topology: TopologySpec):
        torch.manual_seed(0)
        model = topology.build_model()
        quantum_layer = topology.raw_quantum_layer(model)

        generator = torch.Generator().manual_seed(1)
        x = torch.rand(10, N_QUBITS, generator=generator) * 2 * torch.pi
        raw_output = quantum_layer(x)

        assert torch.isfinite(raw_output).all()
        assert torch.all(raw_output >= -1.0 - 1e-6)
        assert torch.all(raw_output <= 1.0 + 1e-6)

    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_full_model_output_is_a_valid_probability(self, topology: TopologySpec):
        """Every topology's readout + Sigmoid maps to [0, 1]."""
        torch.manual_seed(0)
        model = topology.build_model()
        model.eval()
        x = torch.rand(6, N_QUBITS) * 2 * torch.pi
        with torch.no_grad():
            output = model(x)

        assert output.shape == (6, 1)
        assert torch.isfinite(output).all()
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_rejects_wrong_input_width(self, topology: TopologySpec):
        model = topology.build_model()
        with pytest.raises(ValueError):
            model(torch.rand(2, N_QUBITS + 1))


# --------------------------------------------------------------------------- #
# 2. Unitary preservation
# --------------------------------------------------------------------------- #

class TestUnitaryPreservation:
    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_probabilities_sum_to_one(self, topology: TopologySpec):
        """A bug that accidentally introduced a non-unitary operation (a
        mid-circuit measurement, manual renormalization, a non-unitary
        channel) would show up here as `probs.sum() != 1` — this is the
        direct empirical signature of a unitary, probability-conserving
        circuit, independent of the (also-checked) expectation-value
        bounds above."""
        rng = np.random.default_rng(42)
        probs_qnode = topology.probs_qnode_builder(N_QUBITS, N_LAYERS)
        weight_args = topology.weight_args_builder(N_QUBITS, N_LAYERS, rng)
        inputs = torch.from_numpy(rng.uniform(0, 2 * np.pi, size=N_QUBITS)).float()

        probs = probs_qnode(inputs, *weight_args)
        probs_np = np.asarray(probs.detach() if hasattr(probs, "detach") else probs)

        assert probs_np.shape == (2 ** N_QUBITS,)
        assert np.all(probs_np >= -1e-8), "Negative 'probability' — not a valid unitary circuit."
        assert abs(float(probs_np.sum()) - 1.0) < 1e-6

    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    @pytest.mark.parametrize("trial_seed", [0, 1, 2])
    def test_unitary_preservation_holds_across_random_weights(self, topology: TopologySpec, trial_seed: int):
        rng = np.random.default_rng(trial_seed)
        probs_qnode = topology.probs_qnode_builder(N_QUBITS, N_LAYERS)
        weight_args = topology.weight_args_builder(N_QUBITS, N_LAYERS, rng)
        inputs = torch.from_numpy(rng.uniform(0, 2 * np.pi, size=N_QUBITS)).float()

        probs = probs_qnode(inputs, *weight_args)
        probs_np = np.asarray(probs.detach() if hasattr(probs, "detach") else probs)
        assert abs(float(probs_np.sum()) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# 3. No hardware/cloud quantum backend
# --------------------------------------------------------------------------- #

class TestNoHardwareQuantumBackend:
    def test_configured_device_is_the_local_simulator(self):
        """Every topology is built from `config.quantum.device` — asserting
        it's `default.qubit` (PennyLane's local, exact state-vector
        simulator) here covers all three at once and fails loudly if
        someone ever repoints `config.yaml` at a cloud/hardware backend
        without updating this test suite's assumptions."""
        assert _config.quantum.device == "default.qubit"

    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_quantum_layer_device_is_default_qubit(self, topology: TopologySpec):
        model = topology.build_model()
        quantum_layer = topology.raw_quantum_layer(model)
        device_name = quantum_layer.qnode.device.short_name if hasattr(quantum_layer.qnode.device, "short_name") else type(quantum_layer.qnode.device).__name__
        assert "default.qubit" in device_name or "DefaultQubit" in device_name


# --------------------------------------------------------------------------- #
# 4. Determinism
# --------------------------------------------------------------------------- #

class TestDeterminism:
    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_repeated_forward_passes_are_bit_identical(self, topology: TopologySpec):
        torch.manual_seed(7)
        model = topology.build_model()
        model.eval()
        x = torch.rand(4, N_QUBITS) * 2 * torch.pi

        with torch.no_grad():
            output_a = model(x)
            output_b = model(x)

        torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)

    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_probs_qnode_is_deterministic_given_fixed_weights(self, topology: TopologySpec):
        rng = np.random.default_rng(123)
        probs_qnode = topology.probs_qnode_builder(N_QUBITS, N_LAYERS)
        weight_args = topology.weight_args_builder(N_QUBITS, N_LAYERS, rng)
        inputs = torch.from_numpy(rng.uniform(0, 2 * np.pi, size=N_QUBITS)).float()

        probs_a = np.asarray(probs_qnode(inputs, *weight_args))
        probs_b = np.asarray(probs_qnode(inputs, *weight_args))
        np.testing.assert_array_equal(probs_a, probs_b)


# --------------------------------------------------------------------------- #
# 5. Gradient flow (each ansatz must be trainable, not just forward-correct)
# --------------------------------------------------------------------------- #

class TestGradientFlow:
    @pytest.mark.parametrize("topology", TOPOLOGIES, ids=TOPOLOGY_IDS)
    def test_backward_populates_gradients_on_every_trainable_parameter(self, topology: TopologySpec):
        torch.manual_seed(3)
        model = topology.build_model()
        model.train()

        x = torch.rand(4, N_QUBITS) * 2 * torch.pi
        output = model(x)
        output.sum().backward()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert trainable_params, "Expected at least one trainable parameter."
        for param in trainable_params:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
