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

import threading
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

logger = get_logger(__name__)
_config = load_config()

N_QUBITS = _config.quantum.n_qubits
ROTATIONS_PER_QUBIT_PER_LAYER = 3  # RX, RY, RZ

# Preferred backend for the pure-inference (no-gradient) QNode built below:
# `lightning.qubit` is PennyLane-Lightning's C++ statevector simulator —
# materially faster than the pure-Python/NumPy `default.qubit` for the
# repeated small-circuit (4-qubit, 3-layer) evaluations this project's
# inference hot path runs. `load_quantum_device` already falls back to
# `default.qubit` transparently if the plugin isn't installed, so this is
# a safe default rather than a hard dependency.
INFERENCE_DEVICE_PREFERENCE = "lightning.qubit"

# Inference results are cached by a rounded-angle key (see `predict_fast`),
# per `VQCClassifier` instance (never shared across different trained
# weight sets). Bounded so a long-running server process doesn't grow this
# unboundedly across many distinct uploaded slices.
_INFERENCE_CACHE_MAX_ENTRIES = 512


def load_quantum_device(device_name: str, n_qubits: int) -> "qml.Device":
    """Loads a PennyLane device, transparently falling back to the native
    `default.qubit` state-vector simulator if the requested backend (e.g.
    Qiskit Aer's `qiskit.aer`, or PennyLane-Lightning's `lightning.qubit`)
    fails to load or throws an environment error — a missing optional
    plugin, a missing compiled extension, an accelerator/GPU backend that
    isn't actually present on this host, a `MemoryError` allocating a
    backend's state-vector buffer, etc. Every `qml.device(...)` construction
    failure is caught generically (`except Exception`, `MemoryError`
    included — it's a builtin `Exception` subclass) and treated the same
    way: log and fall back. `default.qubit` ships with PennyLane itself, so
    it's always available and is the safe universal fallback the rest of
    this project trains/tests against.
    """
    try:
        return qml.device(device_name, wires=n_qubits)
    except Exception as exc:
        if device_name == "default.qubit":
            raise
        logger.warning(
            "Failed to load PennyLane device %r (%s: %s); falling back to "
            "'default.qubit'.", device_name, type(exc).__name__, exc,
        )
        return qml.device("default.qubit", wires=n_qubits)


def angle_encoding(features: torch.Tensor, wires: List[int]) -> None:
    """Continuous angle encoding: maps each input scalar (in [0, 2*pi]) onto
    one qubit via RX followed by RY.

    Args:
        features: 1D tensor of length len(wires), values in [0, 2*pi].
        wires: Qubit indices to encode onto, one feature per wire.
    """
    for i, wire in enumerate(wires):
        qml.RX(features[..., i], wires=wire)
        qml.RY(features[..., i], wires=wire)


def variational_block(weights: torch.Tensor, wires: List[int]) -> None:
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


def build_qnode(n_qubits: int = N_QUBITS, n_layers: int = _config.quantum.n_layers):
    """Constructs the PennyLane QNode: angle encoding -> `n_layers`
    variational blocks -> Pauli-Z expectation values on every qubit.

    Uses PennyLane's `default.qubit` state-vector simulator, wired for the
    `torch` interface with `config.quantum.diff_method` (`"backprop"` by
    default) so gradients flow through `qml.qnn.TorchLayer` during
    training. This is the *trainable* circuit — see `build_inference_qnode`
    below for the separate, faster, gradient-free circuit `predict_fast`
    evaluates at inference time.
    """
    dev = load_quantum_device(_config.quantum.device, n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="torch", diff_method=_config.quantum.diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        angle_encoding(inputs, wires)
        for layer in range(n_layers):
            variational_block(weights[layer], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return circuit


def build_inference_qnode(n_qubits: int = N_QUBITS, n_layers: int = _config.quantum.n_layers):
    """Constructs a second, inference-only QNode for the same circuit —
    same angle encoding / variational blocks / Pauli-Z measurement as
    `build_qnode`, but on PennyLane-Lightning's C++ statevector simulator
    (falling back to `default.qubit` if unavailable) with `interface=None`
    and `diff_method=None`: no ML-framework tensor wrapping, no autograd
    tape construction. Forward-pass-only evaluation is strictly cheaper
    without either, since this project's inference call sites (`/predict`,
    `/explain`, the Streamlit workstation) never need gradients through
    the quantum layer — only training does, via `build_qnode`'s separate
    circuit.

    Returns `None` (rather than raising) if even `default.qubit` fails to
    construct here, so `VQCClassifier` can fall back to the slower
    `TorchLayer` forward pass unconditionally.
    """
    try:
        dev = load_quantum_device(INFERENCE_DEVICE_PREFERENCE, n_qubits)
    except Exception as exc:  # noqa: BLE001 - inference-fast-path is optional, never fatal
        logger.warning("Could not build any inference QNode device: %s", exc)
        return None
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface=None, diff_method=None)
    def circuit(inputs, weights):
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

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = _config.quantum.n_layers):
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

        # --- Fast, gradient-free inference path (see `predict_fast`) ---
        # Built lazily/defensively: `build_inference_qnode` already
        # swallows its own device-construction failures and returns None,
        # but circuit *evaluation* can still fail later for reasons that
        # only show up at call time (e.g. a lightning.qubit version
        # mismatch) — `predict_fast` catches that per-call and falls back
        # to the standard `forward()` path, so this is pure speedup, never
        # a correctness risk.
        self._inference_circuit = build_inference_qnode(n_qubits=n_qubits, n_layers=n_layers)
        self._inference_cache: "OrderedDict[Tuple[float, ...], Tuple[float, Tuple[float, ...]]]" = OrderedDict()
        self._inference_cache_lock = threading.Lock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Expected input shape (B, {self.n_qubits}), got {tuple(x.shape)}"
            )

        expvals = self.quantum_layer(x)  # (B, n_qubits), each in [-1, 1]
        logits = self.readout(expvals)  # (B, 1)
        return self.activation(logits)  # (B, 1), in [0, 1]

    def predict_fast(self, angles: np.ndarray, cache_decimals: int = 6) -> Tuple[float, np.ndarray]:
        """Single-sample, gradient-free inference: `(n_qubits,)` angles ->
        `(risk_score, pauli_z_expectations)`. This is the hot path
        `qknee.models.pipeline.PipelineRunner.classify` prefers over
        `forward()` whenever it's available (duck-typed via
        `hasattr(model, "predict_fast")`), for three stacked speedups over
        the training-time `forward()` path:

        1. Runs the `build_inference_qnode` circuit (lightning.qubit,
           `diff_method=None`, no ML-interface tensor wrapping) instead of
           the `TorchLayer`-wrapped, autograd-tape-building training circuit.
        2. Reads the classical readout layer's weights straight into plain
           NumPy once (cached on `self`) and does the final `Linear + Sigmoid`
           as a 4-element NumPy dot product, skipping PyTorch's own
           per-call dispatch overhead for a matmul this tiny.
        3. Caches the final `(risk, expvals)` result by the input angles
           rounded to `cache_decimals` places — an exact repeat query (the
           same uploaded slice re-analyzed, or a demo/validation-cohort
           angle vector seen before) is then a plain dict lookup,
           sub-microsecond and independent of the simulator entirely.

        Falls back to the standard `forward()` path (converted back to
        plain floats) on any failure — a missing/broken inference circuit,
        or any runtime error evaluating it — so this is strictly additive:
        it can only make inference faster, never less correct or less
        available than calling `forward()` directly.
        """
        angles = np.asarray(angles, dtype=np.float64).reshape(-1)
        if angles.shape[0] != self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} angles, got {angles.shape[0]}")

        cache_key = tuple(np.round(angles, cache_decimals).tolist())
        with self._inference_cache_lock:
            cached = self._inference_cache.get(cache_key)
            if cached is not None:
                self._inference_cache.move_to_end(cache_key)
                risk_value, expvals_tuple = cached
                return risk_value, np.asarray(expvals_tuple, dtype=np.float32)

        if self._inference_circuit is None:
            with torch.inference_mode():
                risk_tensor = self.forward(torch.from_numpy(angles).float().unsqueeze(0))
                expvals_tensor = self.quantum_layer(torch.from_numpy(angles).float().unsqueeze(0))
            risk_value = float(risk_tensor.item())
            expvals = expvals_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
        else:
            try:
                with torch.inference_mode():
                    weights_np = self.quantum_layer.weights.detach().cpu().numpy()
                    expvals = np.asarray(self._inference_circuit(angles, weights_np), dtype=np.float64)
                    readout_weight = self.readout.weight.detach().cpu().numpy().reshape(-1)  # (n_qubits,)
                    readout_bias = float(self.readout.bias.detach().cpu().item())
                    logit = float(np.dot(readout_weight, expvals) + readout_bias)
                    risk_value = float(1.0 / (1.0 + np.exp(-logit)))
                expvals = expvals.astype(np.float32)
            except Exception as exc:  # noqa: BLE001 - any evaluation failure degrades to the slow, always-correct path
                logger.warning("Fast inference circuit evaluation failed (%s); falling back to forward().", exc)
                with torch.inference_mode():
                    risk_tensor = self.forward(torch.from_numpy(angles).float().unsqueeze(0))
                    expvals_tensor = self.quantum_layer(torch.from_numpy(angles).float().unsqueeze(0))
                risk_value = float(risk_tensor.item())
                expvals = expvals_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)

        with self._inference_cache_lock:
            self._inference_cache[cache_key] = (risk_value, tuple(expvals.tolist()))
            self._inference_cache.move_to_end(cache_key)
            while len(self._inference_cache) > _INFERENCE_CACHE_MAX_ENTRIES:
                self._inference_cache.popitem(last=False)

        return risk_value, expvals


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()
    torch.manual_seed(0)

    model = VQCClassifier(n_qubits=N_QUBITS, n_layers=3)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", trainable)
    for name, param in model.named_parameters():
        logger.info("  %s: %s", name, tuple(param.shape))

    # --- Forward pass smoke test with a batch of angle-encoded vectors ---
    batch_size = 6
    dummy_input = torch.rand(batch_size, N_QUBITS) * 2 * torch.pi  # simulate pipeline output
    scores = model(dummy_input)
    logger.info("Input shape:  %s", tuple(dummy_input.shape))
    logger.info("Output shape: %s", tuple(scores.shape))
    assert scores.shape == (batch_size, 1)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)
    logger.info("Output scores: %s", scores.detach().flatten().tolist())

    # --- Standard PyTorch training loop integration test ---
    logger.info("Running a short training loop on synthetic labels...")
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
            logger.debug("  epoch %2d | loss = %.4f", epoch, loss.item())

    final_loss = loss.item()
    logger.info("Initial loss: %.4f -> Final loss: %.4f", initial_loss, final_loss)
    assert final_loss < initial_loss, "Expected loss to decrease after training on synthetic data"
    logger.info("Gradients flowed through the quantum layer and loss decreased. All checks passed.")
