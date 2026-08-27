"""
4-qubit Variational Quantum Classifier (VQC) for ACL/meniscal tear risk
scoring, built with PennyLane and wrapped as a PyTorch nn.Module.

Pipeline position: consumes the (1, 4) angle-encoded vectors produced by
`pipeline.MRIQuantumPipeline.extract_quantum_features` (values in [0, 2*pi])
and outputs a normalized binary classification score in [0, 1].

Architecture:
    1. Angle Encoding    - each of the 4 input scalars is encoded onto its
                           own qubit via RX then RY rotations.
    2. Variational block - `n_layers` repeats of per-qubit trainable
                           RX/RY/RZ rotations followed by a ring of CNOT
                           entangling gates.
    3. Measurement       - Pauli-Z expectation value on each of the 4
                           qubits, each in [-1, 1].
    4. Classical readout - a single trainable Linear(4, 1) + Sigmoid maps
                           the 4 expectation values to one risk score in
                           [0, 1], so the circuit's 4 measurements are
                           combined into one classification output.

The quantum circuit is wrapped via `qml.qnn.TorchLayer`, so its rotation
parameters (`self.q_weights` inside the layer) are ordinary
`torch.nn.Parameter`s and train with any standard PyTorch optimizer
(Adam, SGD, ...) alongside the classical readout layer.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 4
ROTATIONS_PER_QUBIT_PER_LAYER = 3  # RX, RY, RZ


def angle_encoding(features: torch.Tensor, wires: list[int]) -> None:
    """Continuous angle encoding: maps each input scalar (in [0, 2*pi]) onto
    one qubit via RX followed by RY.

    Args:
        features: 1D tensor of length len(wires), values in [0, 2*pi].
        wires: Qubit indices to encode onto, one feature per wire.
    """
    for i, wire in enumerate(wires):
        qml.RX(features[..., i], wires=wire)
        qml.RY(features[..., i], wires=wire)


def variational_block(weights: torch.Tensor, wires: list[int]) -> None:
    """One trainable variational layer: per-qubit RX/RY/RZ rotations
    followed by a ring of CNOT entangling gates (wire i -> wire i+1, with
    the last wire wrapping back to the first).

    Args:
        weights: Tensor of shape (len(wires), 3) — one (rx, ry, rz) triple
            of trainable angles per qubit for this layer.
        wires: Qubit indices this layer acts on.
    """
    for i, wire in enumerate(wires):
        qml.RX(weights[..., i, 0], wires=wire)
        qml.RY(weights[..., i, 1], wires=wire)
        qml.RZ(weights[..., i, 2], wires=wire)

    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def build_qnode(n_qubits: int = N_QUBITS, n_layers: int = 3):
    """Constructs the PennyLane QNode: angle encoding -> `n_layers`
    variational blocks -> Pauli-Z expectation values on every qubit.

    Uses PennyLane's `default.qubit` state-vector simulator.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        angle_encoding(inputs, wires)
        for layer in range(n_layers):
            variational_block(weights[layer], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return circuit


class VQCClassifier(nn.Module):
    """4-qubit VQC + classical readout for binary tear-risk classification.

    Args:
        n_qubits: Number of qubits / input features (fixed at 4 by the
            feature-reduction pipeline upstream, but left configurable).
        n_layers: Number of variational block repetitions.

    Forward:
        x: (B, n_qubits) tensor, values in [0, 2*pi].
        returns: (B, 1) tensor, values in [0, 1] — probability-like risk score.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        circuit = build_qnode(n_qubits=n_qubits, n_layers=n_layers)
        weight_shapes = {"weights": (n_layers, n_qubits, ROTATIONS_PER_QUBIT_PER_LAYER)}

        # qml.qnn.TorchLayer turns the QNode's `weights` argument into a
        # torch.nn.Parameter, registered under this module for autograd/optim.
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Combines the 4 Pauli-Z expectation values into one risk score.
        self.readout = nn.Linear(n_qubits, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Expected input shape (B, {self.n_qubits}), got {tuple(x.shape)}"
            )

        expvals = self.quantum_layer(x)  # (B, n_qubits), each in [-1, 1]
        logits = self.readout(expvals)  # (B, 1)
        return self.activation(logits)  # (B, 1), in [0, 1]


if __name__ == "__main__":
    torch.manual_seed(0)

    model = VQCClassifier(n_qubits=N_QUBITS, n_layers=3)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")
    for name, param in model.named_parameters():
        print(f"  {name}: {tuple(param.shape)}")

    # --- Forward pass smoke test with a batch of angle-encoded vectors ---
    batch_size = 6
    dummy_input = torch.rand(batch_size, N_QUBITS) * 2 * torch.pi  # simulate pipeline output
    scores = model(dummy_input)
    print(f"\nInput shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(scores.shape)}")
    assert scores.shape == (batch_size, 1)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)
    print(f"Output scores: {scores.detach().flatten().tolist()}")

    # --- Standard PyTorch training loop integration test ---
    print("\nRunning a short training loop on synthetic labels...")
    dummy_labels = torch.randint(0, 2, (batch_size, 1)).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()

    initial_loss = None
    for epoch in range(20):
        optimizer.zero_grad()
        predictions = model(dummy_input)
        loss = loss_fn(predictions, dummy_labels)
        loss.backward()
        optimizer.step()

        if epoch == 0:
            initial_loss = loss.item()
        if epoch % 5 == 0 or epoch == 19:
            print(f"  epoch {epoch:2d} | loss = {loss.item():.4f}")

    final_loss = loss.item()
    print(f"\nInitial loss: {initial_loss:.4f} -> Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Expected loss to decrease after training on synthetic data"
    print("Gradients flowed through the quantum layer and loss decreased. All checks passed.")
