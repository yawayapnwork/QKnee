"""
4-qubit Variational Quantum Classifier (VQC) using PennyLane's
`StronglyEntanglingLayers` template, wrapped as a PyTorch `nn.Module`.

This is a standalone, single-target-qubit-readout variant of the Q-Knee VQC
(compare `vqc_classifier.py`, which measures all 4 qubits and combines them
with a classical readout layer). Here, the circuit itself returns one raw
Pauli-Z expectation value in [-1, 1] — no classical post-processing — which
is the more common "textbook" VQC shape when the entangling block is a
PennyLane template rather than a hand-rolled layer.

Architecture:
    1. Device        - PennyLane `default.qubit`, 4 wires (state-vector
                        simulator; exact, noiseless — appropriate for
                        classical prototyping ahead of real NISQ hardware).
    2. Encoding       - Continuous angle encoding: each of the 4 input
                        scalars (expected in [0, 2*pi]) is loaded via RX
                        then RY on its own qubit.
    3. Entanglement   - `qml.StronglyEntanglingLayers`: `n_layers` repeats
                        of an arbitrary single-qubit rotation (`Rot`, 3
                        trainable angles per qubit) followed by a ring of
                        CNOTs, modeling non-linear cross-qubit (i.e.
                        cross-spatial-feature) interactions.
    4. Measurement    - Pauli-Z expectation value of exactly one target
                        qubit, in [-1, 1].
    5. PyTorch wiring - The QNode is wrapped with `qml.qnn.TorchLayer`, so
                        its `weights` become a `torch.nn.Parameter` and the
                        whole thing drops into a standard optimizer loop.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pennylane as qml
import torch
import torch.nn as nn

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

logger = get_logger(__name__)
_config = load_config()

N_QUBITS = _config.quantum.n_qubits
DEFAULT_N_LAYERS = _config.quantum.n_layers
DEFAULT_TARGET_QUBIT = 0


def angle_encoding(features: torch.Tensor, wires: List[int]) -> None:
    """Continuous angle encoding: each input scalar (in [0, 2*pi]) is loaded
    onto its own qubit via RX followed by RY.

    Args:
        features: 1D tensor of length len(wires).
        wires: Qubit indices to encode onto, one feature per wire.
    """
    for i, wire in enumerate(wires):
        qml.RX(features[..., i], wires=wire)
        qml.RY(features[..., i], wires=wire)


def build_qnode(
    n_qubits: int = N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
    target_qubit: int = DEFAULT_TARGET_QUBIT,
):
    """Constructs the PennyLane QNode: angle encoding -> StronglyEntanglingLayers
    -> single-qubit Pauli-Z expectation value.

    Args:
        n_qubits: Number of wires (and input features).
        n_layers: Depth of the StronglyEntanglingLayers block.
        target_qubit: Which qubit's Pauli-Z expectation value to return.

    Returns:
        A `qml.QNode` with signature `circuit(inputs, weights) -> scalar`,
        using PennyLane's `default.qubit` simulator and the Torch interface
        with backprop-based differentiation.
    """
    if not 0 <= target_qubit < n_qubits:
        raise ValueError(f"target_qubit must be in [0, {n_qubits - 1}], got {target_qubit}")

    device = qml.device(_config.quantum.device, wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(device, interface="torch", diff_method=_config.quantum.diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        angle_encoding(inputs, wires)
        qml.StronglyEntanglingLayers(weights, wires=wires)
        return qml.expval(qml.PauliZ(target_qubit))

    return circuit


def weight_shape(n_qubits: int = N_QUBITS, n_layers: int = DEFAULT_N_LAYERS) -> Tuple[int, int, int]:
    """Returns the `(n_layers, n_qubits, 3)` shape PennyLane's
    `StronglyEntanglingLayers` expects for its trainable weight tensor
    (3 rotation angles per qubit per layer)."""
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)


class VQCModel(nn.Module):
    """4-qubit Variational Quantum Classifier: angle encoding ->
    StronglyEntanglingLayers -> single-qubit Pauli-Z expectation value,
    wrapped as a trainable `nn.Module` via `qml.qnn.TorchLayer`.

    Args:
        n_qubits: Number of qubits / input features.
        n_layers: Depth of the entangling block.
        target_qubit: Which qubit is measured for the output.
        weight_init_range: (low, high) uniform range used to initialize the
            circuit's rotation-angle weights (radians). PennyLane's default
            TorchLayer initialization samples uniformly in [0, 2*pi]; this
            is exposed explicitly here to make initialization reproducible
            and tunable, per "model parameter initialization".

    Forward:
        x: (B, n_qubits) tensor, values expected in [0, 2*pi].
        returns: (B,) tensor, raw Pauli-Z expectation values in [-1, 1].
            (Not a probability — apply e.g. `(1 + x) / 2` or a classical
            readout layer downstream if a [0, 1] score is needed.)
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        target_qubit: int = DEFAULT_TARGET_QUBIT,
        weight_init_range: Tuple[float, float] = (0.0, 2 * torch.pi),
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.target_qubit = target_qubit

        if seed is not None:
            torch.manual_seed(seed)

        circuit = build_qnode(n_qubits=n_qubits, n_layers=n_layers, target_qubit=target_qubit)
        shape = weight_shape(n_qubits=n_qubits, n_layers=n_layers)

        low, high = weight_init_range

        def _init_weights(tensor: torch.Tensor) -> torch.Tensor:
            return nn.init.uniform_(tensor, a=low, b=high)

        self.quantum_layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes={"weights": shape},
            init_method={"weights": _init_weights},
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Expected input shape (B, {self.n_qubits}), got {tuple(x.shape)}"
            )
        return self.quantum_layer(x)


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()
    torch.manual_seed(0)

    model = VQCModel(n_qubits=N_QUBITS, n_layers=DEFAULT_N_LAYERS, target_qubit=0, seed=42)

    logger.info("Device: %s, %d wires", _config.quantum.device, N_QUBITS)
    logger.info("Entangling block: StronglyEntanglingLayers, depth=%d", DEFAULT_N_LAYERS)
    logger.info("Trainable weight tensor shape: %s", tuple(model.quantum_layer.weights.shape))
    logger.info(
        "Weight init range: [0, 2*pi] -> sample min=%.3f, max=%.3f",
        model.quantum_layer.weights.min(), model.quantum_layer.weights.max(),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total trainable parameters: %d", trainable)

    # --- Forward pass smoke test ---
    batch_size = 5
    dummy_input = torch.rand(batch_size, N_QUBITS) * 2 * torch.pi
    output = model(dummy_input)

    logger.info("Input shape:  %s", tuple(dummy_input.shape))
    logger.info("Output shape: %s", tuple(output.shape))
    assert output.shape == (batch_size,)
    assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
    logger.info("Output (Pauli-Z expvals on qubit %d): %s", model.target_qubit, output.detach().tolist())

    # --- Standard PyTorch training loop integration test ---
    # Raw PauliZ expectation values live in [-1, 1], not [0, 1], so a
    # regression loss (MSE) against a [-1, 1] target is the natural fit for
    # this variant's un-post-processed output.
    logger.info("Running a short training loop on synthetic targets...")
    dummy_targets = torch.empty(batch_size).uniform_(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    loss_fn = nn.MSELoss()

    initial_loss = None
    for epoch in range(20):
        optimizer.zero_grad()
        predictions = model(dummy_input)
        loss = loss_fn(predictions, dummy_targets)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        if epoch % 5 == 0 or epoch == 19:
            logger.debug("  epoch %2d | loss = %.4f", epoch, loss.item())

    final_loss = loss.item()
    logger.info("Initial loss: %.4f -> Final loss: %.4f", initial_loss, final_loss)
    assert final_loss < initial_loss, "Expected loss to decrease after training on synthetic data"
    logger.info("Gradients flowed through the quantum layer and loss decreased. All checks passed.")
