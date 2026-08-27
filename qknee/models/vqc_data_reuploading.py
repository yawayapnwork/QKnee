"""
Data-Re-Uploading Variational Quantum Circuit (VQC) for ACL/meniscal tear
risk scoring — Pérez-Salinas et al. (2020), "Data re-uploading for a
universal quantum classifier"-style ansatz that re-encodes the classical
input at *every* variational layer, instead of once up front.

Why re-upload: a single angle-encoding pass (`vqc.VQCClassifier`) can only
apply one rotation per qubit derived from the raw input before the
entangling block takes over, so the circuit's expressivity in the input
is limited by the encoding gate count. Repeating the encode -> vary cycle
`n_layers` times — each time through a *different*, independently
trainable affine remap of the same input (`w * x + b`) — lets the circuit
approximate a much richer family of non-linear functions of the classical
input at the same qubit count, without adding wires or depending on
additional ancillas.

Architecture (per layer, repeated `n_layers` times):
    1. Re-uploading encoding - each of the `n_qubits` input scalars is
       affinely remapped (trainable per-qubit scale + bias) and loaded via
       RX then RY. Reusing the *same* raw input across layers, combined
       with different learned (scale, bias) pairs, is what "re-uploading"
       means here.
    2. Variational block     - per-qubit trainable RX/RY/RZ rotations
       followed by a ring of CNOT entangling gates (same shape/convention
       as `vqc.variational_block`).
    3. Measurement            - Pauli-Z expectation value on every qubit,
       each in [-1, 1].
    4. Classical readout      - a trainable Linear(n_qubits, 1) + Sigmoid
       combines the qubit measurements into one risk score in [0, 1], so
       this module is a drop-in replacement for `vqc.VQCClassifier`
       wherever a `(B, n_qubits) -> (B, 1)` classifier is expected.

Both `enc_weights` (the re-uploading affine encoders) and `var_weights`
(the entangling-block rotations) are wrapped via `qml.qnn.TorchLayer`, so
they become ordinary `torch.nn.Parameter`s: forward/backward passes go
through PennyLane's `backprop` differentiation on `default.qubit` exactly
like `VQCClassifier`, and this module trains with any standard PyTorch
optimizer alongside the rest of the pipeline — full autograd
compatibility, no manual parameter-shift wiring required.
"""

from __future__ import annotations

from typing import List, Tuple

import pennylane as qml
import torch
import torch.nn as nn

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

logger = get_logger(__name__)
_config = load_config()

N_QUBITS = _config.quantum.n_qubits
DEFAULT_N_LAYERS = _config.quantum.n_layers
ROTATIONS_PER_QUBIT_PER_LAYER = 3         # RX, RY, RZ in the variational block
ENCODING_PARAMS_PER_QUBIT_PER_LAYER = 4   # (rx_scale, rx_bias, ry_scale, ry_bias)


def reuploading_encoding(features: torch.Tensor, enc_weights: torch.Tensor, wires: List[int]) -> None:
    """One trainable re-uploading encoding pass: each input feature `x_i`
    is remapped through a trainable affine transform (`scale * x_i + bias`)
    before being loaded via RX then RY — the data re-uploading trick's key
    departure from single-shot angle encoding (`vqc.angle_encoding`), which
    loads `x_i` unmodified.

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
    """One trainable entangling layer: per-qubit RX/RY/RZ rotations
    followed by a ring of CNOT gates (wire i -> wire i+1, wrapping around).
    Structurally identical to `vqc.variational_block`; re-declared here so
    this module has no dependency on `vqc.py`'s single-shot-encoding
    circuit.

    Args:
        weights: (len(wires), 3) tensor — one (rx, ry, rz) triple of
            trainable angles per qubit for this layer.
        wires: Qubit indices this layer acts on.
    """
    for i, wire in enumerate(wires):
        qml.RX(weights[..., i, 0], wires=wire)
        qml.RY(weights[..., i, 1], wires=wire)
        qml.RZ(weights[..., i, 2], wires=wire)

    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


def build_qnode(n_qubits: int = N_QUBITS, n_layers: int = DEFAULT_N_LAYERS):
    """Constructs the data-re-uploading QNode:

        for layer in range(n_layers):
            re-upload the classical input (trainable affine encoding)
            apply a trainable variational entangling block
        measure Pauli-Z on every qubit

    Uses PennyLane's `default.qubit` state-vector simulator with the Torch
    interface and `config.quantum.diff_method` (typically `"backprop"`),
    so gradients flow end-to-end through both `enc_weights` and
    `var_weights` via ordinary `.backward()` calls.

    Returns:
        A `qml.QNode` with signature
        `circuit(inputs, enc_weights, var_weights) -> List[scalar]`
        (one Pauli-Z expectation value per qubit).
    """
    device = qml.device(_config.quantum.device, wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(device, interface="torch", diff_method=_config.quantum.diff_method)
    def circuit(inputs: torch.Tensor, enc_weights: torch.Tensor, var_weights: torch.Tensor):
        for layer in range(n_layers):
            reuploading_encoding(inputs, enc_weights[layer], wires)
            variational_block(var_weights[layer], wires)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return circuit


def weight_shapes(n_qubits: int = N_QUBITS, n_layers: int = DEFAULT_N_LAYERS) -> dict:
    """Returns the `{"enc_weights": ..., "var_weights": ...}` shape dict
    `qml.qnn.TorchLayer` expects for this circuit's two trainable tensors."""
    return {
        "enc_weights": (n_layers, n_qubits, ENCODING_PARAMS_PER_QUBIT_PER_LAYER),
        "var_weights": (n_layers, n_qubits, ROTATIONS_PER_QUBIT_PER_LAYER),
    }


def _init_enc_weights(tensor: torch.Tensor) -> torch.Tensor:
    """Initializes each layer's re-uploading affine encoder near the
    identity map (`scale=1, bias=0`) — i.e. layer 0 starts out behaving
    like plain angle encoding, and training is free to depart from that as
    it discovers a more useful per-layer remap. In-place, matching
    `qml.qnn.TorchLayer`'s `init_method` contract (mirrors
    `vqc_strongly_entangling.py`'s `_init_weights`).
    """
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


class DataReuploadingVQC(nn.Module):
    """Data-Re-Uploading VQC + classical readout for binary tear-risk
    classification — a drop-in alternative to `vqc.VQCClassifier` with
    higher expressive capacity at matched qubit count and depth, since the
    classical input is re-encoded at every layer instead of once.

    Args:
        n_qubits: Number of qubits / input features.
        n_layers: Number of (re-upload -> variational-block) repetitions.

    Forward:
        x: (B, n_qubits) tensor, values in [0, 2*pi].
        returns: (B, 1) tensor, values in [0, 1] — probability-like risk score.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = DEFAULT_N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        circuit = build_qnode(n_qubits=n_qubits, n_layers=n_layers)
        self.quantum_layer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes=weight_shapes(n_qubits=n_qubits, n_layers=n_layers),
            init_method={"enc_weights": _init_enc_weights, "var_weights": _init_var_weights},
        )

        # Combines the n_qubits Pauli-Z expectation values into one risk score.
        self.readout = nn.Linear(n_qubits, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != self.n_qubits:
            raise ValueError(
                f"Expected input shape (B, {self.n_qubits}), got {tuple(x.shape)}"
            )

        expvals = self.quantum_layer(x)  # (B, n_qubits), each in [-1, 1]
        logits = self.readout(expvals)   # (B, 1)
        return self.activation(logits)   # (B, 1), in [0, 1]


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()
    torch.manual_seed(0)

    model = DataReuploadingVQC(n_qubits=N_QUBITS, n_layers=DEFAULT_N_LAYERS)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Data-Re-Uploading VQC: n_qubits=%d, n_layers=%d, trainable params=%d",
                N_QUBITS, DEFAULT_N_LAYERS, trainable)
    for name, param in model.named_parameters():
        logger.info("  %s: %s", name, tuple(param.shape))

    # --- Forward pass smoke test ---
    batch_size = 6
    dummy_input = torch.rand(batch_size, N_QUBITS) * 2 * torch.pi
    scores = model(dummy_input)
    assert scores.shape == (batch_size, 1)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)
    logger.info("Forward pass OK: input %s -> output %s", tuple(dummy_input.shape), tuple(scores.shape))

    # --- Autograd + training loop integration test ---
    logger.info("Running a short training loop on synthetic labels...")
    dummy_labels = torch.randint(0, 2, (batch_size, 1)).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.BCELoss()

    initial_loss: float = float("nan")
    for epoch in range(20):
        optimizer.zero_grad()
        predictions = model(dummy_input)
        loss = loss_fn(predictions, dummy_labels)
        loss.backward()

        # Confirm gradients actually reached both trainable weight tensors
        # (i.e. re-uploading encoding weights are not silently detached).
        if epoch == 0:
            initial_loss = loss.item()
            assert model.quantum_layer.enc_weights.grad is not None
            assert model.quantum_layer.var_weights.grad is not None
            assert torch.any(model.quantum_layer.enc_weights.grad != 0)

        optimizer.step()
        if epoch % 5 == 0 or epoch == 19:
            logger.debug("  epoch %2d | loss = %.4f", epoch, loss.item())

    final_loss = loss.item()
    logger.info("Initial loss: %.4f -> Final loss: %.4f", initial_loss, final_loss)
    assert final_loss < initial_loss, "Expected loss to decrease after training on synthetic data"
    logger.info("Gradients flowed through both re-uploading and variational weights; loss decreased. All checks passed.")
