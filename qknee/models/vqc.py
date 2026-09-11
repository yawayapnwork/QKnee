"""
4-qubit Variational Quantum Classifier (VQC) for ACL/meniscal tear risk
scoring, built with PennyLane and wrapped as a PyTorch nn.Module.

Pipeline position: consumes the (1, 4) angle-encoded vectors produced by
`pipeline.MRIQuantumPipeline.extract_quantum_features` (values in [0, 2*pi])
and outputs a normalized binary classification score in [0, 1].

One ansatz, selected via the `ansatz` constructor argument — this module
used to be split across two files (`vqc.py` / `vqc_data_reuploading.py`)
with two separate classes; they're consolidated here because the second
ansatz was always a drop-in `(B, n_qubits) -> (B, 1)` alternative sharing
this module's `variational_block`, not an independently-evolving design:

    "angle" (default, judged PRD path):
        1. Angle Encoding    - each of the `n_qubits` input scalars is
                               encoded ONCE, up front, onto its own qubit
                               via RX then RY rotations.
        2. Variational block - `n_layers` repeats of per-qubit trainable
                               RX/RY/RZ rotations followed by a ring of
                               CNOT entangling gates.
        3. Measurement       - Pauli-Z expectation value on each qubit,
                               each in [-1, 1].
        4. Classical readout - a trainable Linear(n_qubits, 1) + Sigmoid
                               maps the expectation values to one risk
                               score in [0, 1].

    "data_reuploading" (exploratory ablation — Pérez-Salinas et al. 2020,
    "Data re-uploading for a universal quantum classifier"):
        Re-encodes the classical input at *every* variational layer
        instead of once. A single angle-encoding pass can only apply one
        rotation per qubit derived from the raw input before the
        entangling block takes over, so the circuit's expressivity in the
        input is limited by the encoding gate count; re-uploading — each
        layer's input pass through a different, independently trainable
        affine remap (`scale * x + bias`) of the SAME raw input — lets the
        circuit approximate a much richer family of non-linear functions
        of the classical input at the same qubit count and depth, at the
        cost of `4 * n_qubits * n_layers` extra trainable encoder
        parameters over the "angle" ansatz. Steps per layer:
            1. Re-uploading encoding - trainable affine remap + RX/RY.
            2. Variational block     - identical to "angle"'s.
        Measurement and classical readout are identical to "angle".

Both ansatzes are wrapped via `qml.qnn.TorchLayer`, so their rotation
parameters are ordinary `torch.nn.Parameter`s and train with any standard
PyTorch optimizer (Adam, SGD, ...) alongside the classical readout layer.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, List, Literal, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

logger = get_logger(__name__)
_config = load_config()

N_QUBITS = _config.quantum.n_qubits
ROTATIONS_PER_QUBIT_PER_LAYER = 3          # RX, RY, RZ in the variational block
ENCODING_PARAMS_PER_QUBIT_PER_LAYER = 4    # (rx_scale, rx_bias, ry_scale, ry_bias) -- "data_reuploading" only

Ansatz = Literal["angle", "data_reuploading"]

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


MAX_FEATURES_PER_QUBIT = 3  # RX, RY, RZ — the 3 independent single-qubit rotation axes;
                            # beyond this, extra rotations about an already-used axis
                            # collapse into the earlier one (RX(a) . RX(b) == RX(a+b))
                            # without an interleaving entangling gate, so they'd add no
                            # information despite consuming an extra classical feature.
_ENCODING_GATES = (qml.RX, qml.RY, qml.RZ)


def angle_encoding(features: torch.Tensor, wires: List[int], features_per_qubit: int = 1) -> None:
    """Continuous angle encoding: loads `features_per_qubit` classical scalars
    (each in [0, 2*pi]) onto each qubit via `features_per_qubit` rotation
    gates, then moves to the next qubit. Used once, up front, by the "angle"
    ansatz — see `reuploading_encoding` for the per-layer alternative the
    "data_reuploading" ansatz uses instead.

    `features_per_qubit=1` (the default) is the original, checkpoint-
    compatible behavior: one scalar per qubit, loaded via RX *and* RY with
    the SAME value (a cheap way to fill more of the Bloch sphere than a
    single rotation would from one input number). `features_per_qubit=2` or
    `3` instead loads that many *distinct* scalars per qubit — one via RX,
    the next via RY, the next (if 3) via RZ — so a fixed `n_qubits` circuit
    can consume more of a PCA-reduced feature vector's variance without
    discarding it at the dimensionality-reduction step (see
    `qknee.models.pca_reducer.QuantumDimReducer` and
    `config.quantum.features_per_qubit`).

    Args:
        features: 1D tensor. Length `len(wires)` when `features_per_qubit=1`
            (legacy); length `len(wires) * features_per_qubit` otherwise,
            laid out as `[qubit0_feat0, qubit0_feat1, ..., qubit1_feat0, ...]`.
            Values in [0, 2*pi].
        wires: Qubit indices to encode onto.
        features_per_qubit: How many distinct classical features (and
            rotation gates) to load per qubit. Must be in `[1, MAX_FEATURES_PER_QUBIT]`.
    """
    if features_per_qubit == 1:
        for i, wire in enumerate(wires):
            qml.RX(features[..., i], wires=wire)
            qml.RY(features[..., i], wires=wire)
        return

    if not (1 <= features_per_qubit <= MAX_FEATURES_PER_QUBIT):
        raise ValueError(
            f"features_per_qubit must be in [1, {MAX_FEATURES_PER_QUBIT}], got {features_per_qubit}"
        )
    for i, wire in enumerate(wires):
        base = i * features_per_qubit
        for g in range(features_per_qubit):
            _ENCODING_GATES[g](features[..., base + g], wires=wire)


def reuploading_encoding(features: torch.Tensor, enc_weights: torch.Tensor, wires: List[int]) -> None:
    """One trainable re-uploading encoding pass: each input feature `x_i`
    is remapped through a trainable affine transform (`scale * x_i + bias`)
    before being loaded via RX then RY — the data re-uploading ansatz's key
    departure from `angle_encoding`, which loads `x_i` unmodified and only
    once. Called once per layer (with that layer's own `enc_weights`) by
    the "data_reuploading" ansatz.

    Args:
        features: (..., n_qubits) tensor, values typically in [0, 2*pi]
            (e.g. `QuantumDimReducer`'s PCA-angle output).
        enc_weights: (n_qubits, 4) tensor of this layer's
            (rx_scale, rx_bias, ry_scale, ry_bias) per qubit.
        wires: Qubit indices to encode onto, one feature per wire.
    """
    for i, wire in enumerate(wires):
        rx_scale = enc_weights[i, 0]
        rx_bias = enc_weights[i, 1]
        ry_scale = enc_weights[i, 2]
        ry_bias = enc_weights[i, 3]
        qml.RX(rx_scale * features[..., i] + rx_bias, wires=wire)
        qml.RY(ry_scale * features[..., i] + ry_bias, wires=wire)


def variational_block(weights: torch.Tensor, wires: List[int]) -> None:
    """One trainable variational layer: per-qubit RX/RY/RZ rotations
    followed by a ring of CNOT entangling gates (wire i -> wire i+1, with
    the last wire wrapping back to the first). Shared by both ansatzes.

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


def weight_shapes_for(ansatz: Ansatz, n_qubits: int, n_layers: int) -> Dict[str, tuple]:
    """Returns the `qml.qnn.TorchLayer` weight-shape dict for `ansatz` —
    one tensor (`weights`) for "angle", two (`enc_weights`, `var_weights`)
    for "data_reuploading"."""
    if ansatz == "angle":
        return {"weights": (n_layers, n_qubits, ROTATIONS_PER_QUBIT_PER_LAYER)}
    if ansatz == "data_reuploading":
        return {
            "enc_weights": (n_layers, n_qubits, ENCODING_PARAMS_PER_QUBIT_PER_LAYER),
            "var_weights": (n_layers, n_qubits, ROTATIONS_PER_QUBIT_PER_LAYER),
        }
    raise ValueError(f"Unknown ansatz {ansatz!r}; expected 'angle' or 'data_reuploading'.")


def _init_enc_weights(tensor: torch.Tensor) -> torch.Tensor:
    """Initializes each layer's re-uploading affine encoder near the
    identity map (`scale=1, bias=0`) — i.e. layer 0 starts out behaving
    like plain angle encoding, and training is free to depart from that as
    it discovers a more useful per-layer remap. In-place, matching
    `qml.qnn.TorchLayer`'s `init_method` contract."""
    with torch.no_grad():
        tensor[..., 0].fill_(1.0)  # rx_scale
        tensor[..., 1].fill_(0.0)  # rx_bias
        tensor[..., 2].fill_(1.0)  # ry_scale
        tensor[..., 3].fill_(0.0)  # ry_bias
    return tensor


def _init_var_weights(tensor: torch.Tensor) -> torch.Tensor:
    """Initializes the variational block's rotation angles uniformly in
    [0, 2*pi] — `qml.qnn.TorchLayer`'s own default for an otherwise
    unspecified weight tensor, made explicit here since `init_method` must
    cover every weight name once any entry is supplied."""
    return nn.init.uniform_(tensor, a=0.0, b=2 * torch.pi)


def build_qnode(
    n_qubits: int = N_QUBITS,
    n_layers: int = _config.quantum.n_layers,
    ansatz: Ansatz = "angle",
    features_per_qubit: int = 1,
):
    """Constructs the PennyLane QNode for `ansatz`:

        "angle":             encode once -> `n_layers` variational blocks
        "data_reuploading":   `n_layers` repeats of (re-encode -> variational block)

    -> Pauli-Z expectation values on every qubit.

    Uses PennyLane's `default.qubit` state-vector simulator, wired for the
    `torch` interface with `config.quantum.diff_method` (`"backprop"` by
    default) so gradients flow through `qml.qnn.TorchLayer` during
    training. This is the *trainable* circuit — see `build_inference_qnode`
    below for the separate, faster, gradient-free circuit `predict_fast`
    evaluates at inference time.

    Returns a QNode with signature `circuit(inputs, weights)` for "angle",
    or `circuit(inputs, enc_weights, var_weights)` for "data_reuploading".
    """
    dev = load_quantum_device(_config.quantum.device, n_qubits)
    wires = list(range(n_qubits))

    if ansatz == "angle":
        @qml.qnode(dev, interface="torch", diff_method=_config.quantum.diff_method)
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            angle_encoding(inputs, wires, features_per_qubit=features_per_qubit)
            for layer in range(n_layers):
                variational_block(weights[layer], wires)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return circuit

    if ansatz == "data_reuploading":
        @qml.qnode(dev, interface="torch", diff_method=_config.quantum.diff_method)
        def circuit(inputs: torch.Tensor, enc_weights: torch.Tensor, var_weights: torch.Tensor):
            for layer in range(n_layers):
                reuploading_encoding(inputs, enc_weights[layer], wires)
                variational_block(var_weights[layer], wires)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return circuit

    raise ValueError(f"Unknown ansatz {ansatz!r}; expected 'angle' or 'data_reuploading'.")


def build_inference_qnode(
    n_qubits: int = N_QUBITS,
    n_layers: int = _config.quantum.n_layers,
    ansatz: Ansatz = "angle",
    features_per_qubit: int = 1,
):
    """Constructs a second, inference-only QNode for the same circuit —
    same gate sequence as `build_qnode` for the given `ansatz`, but on
    PennyLane-Lightning's C++ statevector simulator (falling back to
    `default.qubit` if unavailable) with `interface=None` and
    `diff_method=None`: no ML-framework tensor wrapping, no autograd tape
    construction. Forward-pass-only evaluation is strictly cheaper without
    either, since this project's inference call sites (`/predict`,
    `/explain`) never need gradients through the quantum layer — only
    training does, via `build_qnode`'s separate circuit.

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

    if ansatz == "angle":
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit(inputs, weights):
            angle_encoding(inputs, wires, features_per_qubit=features_per_qubit)
            for layer in range(n_layers):
                variational_block(weights[layer], wires)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return circuit

    if ansatz == "data_reuploading":
        @qml.qnode(dev, interface=None, diff_method=None)
        def circuit(inputs, enc_weights, var_weights):
            for layer in range(n_layers):
                reuploading_encoding(inputs, enc_weights[layer], wires)
                variational_block(var_weights[layer], wires)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return circuit

    raise ValueError(f"Unknown ansatz {ansatz!r}; expected 'angle' or 'data_reuploading'.")


class VQCClassifier(nn.Module):
    """4-qubit VQC + classical readout for binary tear-risk classification.

    Args:
        n_qubits: Number of qubits / input features (fixed at 4 by the
            feature-reduction pipeline upstream, but left configurable).
        n_layers: Number of variational block repetitions.
        ansatz: "angle" (default, judged PRD path — single up-front
            encoding) or "data_reuploading" (exploratory ablation —
            re-encodes the input at every layer; see module docstring).
            Changing this from the default changes which/how many
            parameters `self.quantum_layer` registers, so a checkpoint
            trained under one ansatz cannot be loaded under the other.
        features_per_qubit: How many distinct classical features
            `angle_encoding` loads per qubit (1-3; see `angle_encoding`'s
            docstring). Only supported with `ansatz="angle"` — raises if
            combined with `"data_reuploading"`, which has its own per-layer
            re-encoding scheme. Input width becomes `n_qubits *
            features_per_qubit` instead of `n_qubits`. Defaults to 1
            (checkpoint-compatible legacy behavior).

    Forward:
        x: (B, n_qubits * features_per_qubit) tensor, values in [0, 2*pi].
        returns: (B, 1) tensor, values in [0, 1] — probability-like risk score.
    """

    def __init__(
        self,
        n_qubits: int = N_QUBITS,
        n_layers: int = _config.quantum.n_layers,
        ansatz: Ansatz = "angle",
        features_per_qubit: int = 1,
    ):
        super().__init__()
        if features_per_qubit != 1 and ansatz != "angle":
            raise ValueError(
                f"features_per_qubit={features_per_qubit} is only supported with ansatz='angle' "
                f"(got ansatz={ansatz!r}); 'data_reuploading' has its own per-layer re-encoding scheme."
            )
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz: Ansatz = ansatz
        self.features_per_qubit = features_per_qubit
        self.input_dim = n_qubits * features_per_qubit

        circuit = build_qnode(n_qubits=n_qubits, n_layers=n_layers, ansatz=ansatz, features_per_qubit=features_per_qubit)
        shapes = weight_shapes_for(ansatz, n_qubits, n_layers)

        # qml.qnn.TorchLayer turns the QNode's weight argument(s) into
        # torch.nn.Parameter(s), registered under this module for
        # autograd/optim. Only "data_reuploading" needs a non-default
        # init_method (see _init_enc_weights/_init_var_weights); "angle"'s
        # single `weights` tensor uses TorchLayer's own default (uniform
        # in [0, 2*pi]), preserved exactly as before so existing "angle"
        # checkpoints (acl_vqc.pt, meniscus_vqc.pt, qknee_model.pt) still
        # match this module's parameter names/shapes.
        if ansatz == "data_reuploading":
            self.quantum_layer = qml.qnn.TorchLayer(
                circuit, shapes,
                init_method={"enc_weights": _init_enc_weights, "var_weights": _init_var_weights},
            )
        else:
            self.quantum_layer = qml.qnn.TorchLayer(circuit, shapes)

        # Combines the n_qubits Pauli-Z expectation values into one risk score.
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
        self._inference_circuit = build_inference_qnode(
            n_qubits=n_qubits, n_layers=n_layers, ansatz=ansatz, features_per_qubit=features_per_qubit,
        )
        self._inference_cache: "OrderedDict[Tuple[float, ...], Tuple[float, Tuple[float, ...]]]" = OrderedDict()
        self._inference_cache_lock = threading.Lock()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input shape (B, {self.input_dim}), got {tuple(x.shape)}"
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
        available than calling `forward()` directly. Works identically for
        both ansatzes: the only branch is how many weight tensors are read
        off `self.quantum_layer` and passed to `self._inference_circuit`.
        """
        angles = np.asarray(angles, dtype=np.float64).reshape(-1)
        if angles.shape[0] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} angles, got {angles.shape[0]}")

        cache_key = tuple(np.round(angles, cache_decimals).tolist())
        with self._inference_cache_lock:
            cached = self._inference_cache.get(cache_key)
            if cached is not None:
                self._inference_cache.move_to_end(cache_key)
                risk_value, expvals_tuple = cached
                return risk_value, np.asarray(expvals_tuple, dtype=np.float32)

        if self._inference_circuit is None:
            risk_value, expvals = self._forward_fallback(angles)
        else:
            try:
                with torch.inference_mode():
                    if self.ansatz == "data_reuploading":
                        enc_np = self.quantum_layer.enc_weights.detach().cpu().numpy()
                        var_np = self.quantum_layer.var_weights.detach().cpu().numpy()
                        expvals = np.asarray(self._inference_circuit(angles, enc_np, var_np), dtype=np.float64)
                    else:
                        weights_np = self.quantum_layer.weights.detach().cpu().numpy()
                        expvals = np.asarray(self._inference_circuit(angles, weights_np), dtype=np.float64)
                    readout_weight = self.readout.weight.detach().cpu().numpy().reshape(-1)  # (n_qubits,)
                    readout_bias = float(self.readout.bias.detach().cpu().item())
                    logit = float(np.dot(readout_weight, expvals) + readout_bias)
                    risk_value = float(1.0 / (1.0 + np.exp(-logit)))
                expvals = expvals.astype(np.float32)
            except Exception as exc:  # noqa: BLE001 - any evaluation failure degrades to the slow, always-correct path
                logger.warning("Fast inference circuit evaluation failed (%s); falling back to forward().", exc)
                risk_value, expvals = self._forward_fallback(angles)

        with self._inference_cache_lock:
            self._inference_cache[cache_key] = (risk_value, tuple(expvals.tolist()))
            self._inference_cache.move_to_end(cache_key)
            while len(self._inference_cache) > _INFERENCE_CACHE_MAX_ENTRIES:
                self._inference_cache.popitem(last=False)

        return risk_value, expvals

    def _forward_fallback(self, angles: np.ndarray) -> Tuple[float, np.ndarray]:
        """The always-correct (but slower) path `predict_fast` falls back
        to: the standard `forward()`/`quantum_layer()` TorchLayer calls,
        converted back to plain floats/NumPy."""
        with torch.inference_mode():
            angles_t = torch.from_numpy(angles).float().unsqueeze(0)
            risk_tensor = self.forward(angles_t)
            expvals_tensor = self.quantum_layer(angles_t)
        risk_value = float(risk_tensor.item())
        expvals = expvals_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
        return risk_value, expvals


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()

    for ansatz in ("angle", "data_reuploading"):
        torch.manual_seed(0)
        logger.info("=== ansatz=%s ===", ansatz)
        model = VQCClassifier(n_qubits=N_QUBITS, n_layers=3, ansatz=ansatz)

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
