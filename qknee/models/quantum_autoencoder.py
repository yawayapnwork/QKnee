"""
Exploratory — outside the judged PRD scope (ingestion -> ResNet18 -> PCA -> 4-qubit VQC -> Streamlit UI -> Grad-CAM -> SVM benchmark), which uses `pca_reducer.QuantumDimReducer`'s classical PCA path, not this trainable quantum compressor.

Trainable Quantum Autoencoder (QAE) — Romero, Olson & Aspuru-Guzik (2017),
"Quantum autoencoders for efficient compression of quantum data" — style
compression of ResNet18 embeddings directly in Hilbert space, as a
trainable alternative to `qknee.models.pca_reducer.QuantumDimReducer`'s
classical StandardScaler -> PCA -> MinMaxScaler pipeline.

Why a classical pre-projection is still here: amplitude- or angle-encoding
a raw 512-D ResNet embedding would require one wire per feature (or
log2(512) = 9 wires for amplitude encoding of a normalized vector), and
`default.qubit` state-vector simulation cost is exponential in wire count
— fine at 9-13 wires, intractable well before 512. A small trainable
`nn.Linear` first projects the 512-D embedding down to `n_input_qubits`
classical values; those are angle-encoded onto the circuit's input
register. This projection is trained *jointly* with the quantum encoder
against the same reconstruction objective below, so it isn't a second,
independent classical dimensionality reducer bolted on afterward — it's
the classical half of one trainable compression scheme, analogous to how
a classical autoencoder's first Linear layer is itself learned.

Compression scheme (Romero et al.'s SWAP-test-based QAE, adapted for a
hybrid classical/quantum pipeline):
    1. Encode      - project 512-D -> `n_input_qubits` classical values,
                     angle-encode them onto an `n_input_qubits`-wire
                     register.
    2. Encoder ansatz - a trainable `qml.StronglyEntanglingLayers` block
                     over the full input register — this is the circuit
                     being trained to *compress*: it has no "decoder"
                     half, unlike a classical autoencoder.
    3. Latent/trash split - the first `n_latent_qubits` wires are kept as
                     the compressed representation; the remaining
                     `n_trash_qubits = n_input_qubits - n_latent_qubits`
                     wires are the "trash" register that should end up in
                     a fixed |0...0> product state if compression is
                     lossless.
    4. SWAP test   - a multi-qubit SWAP test (Hadamard -> pairwise CSWAP,
                     each controlled by one shared ancilla -> Hadamard ->
                     measure the ancilla's Pauli-Z expectation) scores the
                     overlap between the trash register and fixed
                     reference qubits prepared in |0...0>. That overlap
                     *is* the reconstruction fidelity — training
                     maximizes it, without ever explicitly decoding back
                     to 512-D, which is what makes this a genuine quantum
                     (not classical-autoencoder-shaped) compression
                     objective.
    5. Latent readout - the `n_latent_qubits` latent wires' Pauli-Z
                     expectation values are rescaled from [-1, 1] to
                     [0, 2*pi], so they're a drop-in replacement for
                     `QuantumDimReducer`'s MinMax-scaled PCA output when
                     fed into a downstream angle-encoding VQC
                     (`vqc.VQCClassifier` / `vqc_data_reuploading.DataReuploadingVQC`).

Everything (`classical_projection` and the QNode's `weights`) is wrapped
in ordinary `nn.Module`/`qml.qnn.TorchLayer` machinery, so both the
reconstruction-fidelity loss and any downstream classification loss can
backprop through this module with standard PyTorch autograd — no manual
parameter-shift wiring, and (unlike `QuantumDimReducer`, which needs the
separate `PCAProjectionLayer` re-implementation to be Grad-CAM-differentiable)
this module is differentiable natively.
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

N_LATENT_QUBITS = _config.quantum.n_qubits   # compressed output size (matches the downstream VQC's n_qubits)
DEFAULT_N_INPUT_QUBITS = 6                    # wires holding the classically pre-projected embedding
DEFAULT_N_LAYERS = _config.quantum.n_layers


def swap_test(ancilla: int, register_a: List[int], register_b: List[int]) -> None:
    """Multi-qubit SWAP test: entangles `ancilla` with pairwise CSWAPs
    between `register_a[i]` and `register_b[i]` for every i, so that

        <Z(ancilla)> = 2 * |<psi_a|psi_b>|^2 - 1

    i.e. the ancilla's Pauli-Z expectation value linearly encodes the
    overlap (fidelity) between the two equal-length registers' joint
    state — `<Z> = 1` for identical states, `<Z> = -1` for orthogonal
    ones. Used here to score how close the "trash" register is to a fixed
    |0...0> reference register.

    Args:
        ancilla: Wire index of the single shared SWAP-test ancilla.
        register_a: Wire indices of the first register (here: trash qubits).
        register_b: Wire indices of the second register (here: reference qubits).
    """
    if len(register_a) != len(register_b):
        raise ValueError(
            f"SWAP test registers must be the same length, got {len(register_a)} vs {len(register_b)}"
        )
    qml.Hadamard(wires=ancilla)
    for wire_a, wire_b in zip(register_a, register_b):
        qml.CSWAP(wires=[ancilla, wire_a, wire_b])
    qml.Hadamard(wires=ancilla)


def build_qae_qnode(
    n_input_qubits: int = DEFAULT_N_INPUT_QUBITS,
    n_latent_qubits: int = N_LATENT_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
) -> Tuple[qml.QNode, int]:
    """Constructs the QAE's QNode.

    Wire layout (`default.qubit`, `n_input_qubits + n_trash_qubits + 1` wires total):
        [0 .. n_latent_qubits-1]                         -> latent qubits (kept; the compressed representation)
        [n_latent_qubits .. n_input_qubits-1]             -> trash qubits (encoder should drive these to |0>)
        [n_input_qubits .. n_input_qubits+n_trash_qubits-1] -> reference qubits, fixed in |0>
        [n_input_qubits + n_trash_qubits]                 -> SWAP-test ancilla

    Returns:
        (circuit, n_wires) where `circuit` has signature
        `circuit(inputs, weights) -> [*latent_expvals, fidelity_expval]`
        (a `list`, not a `tuple` — required for `qml.qnn.TorchLayer` to
        reshape the batched output to `(B, n_latent_qubits + 1)` rather
        than leaving it in a raw per-measurement shape) — `n_latent_qubits`
        Pauli-Z expectation values followed by the SWAP-test ancilla's
        Pauli-Z expectation value. `n_wires` is the total device wire
        count (needed by the caller to size the `default.qubit` device
        consistently, since it's derived from `n_input_qubits`/`n_latent_qubits`).
    """
    if n_latent_qubits >= n_input_qubits:
        raise ValueError(
            f"n_latent_qubits ({n_latent_qubits}) must be < n_input_qubits ({n_input_qubits}) "
            "for the autoencoder to actually compress anything."
        )

    n_trash_qubits = n_input_qubits - n_latent_qubits
    latent_wires = list(range(n_latent_qubits))
    trash_wires = list(range(n_latent_qubits, n_input_qubits))
    reference_wires = list(range(n_input_qubits, n_input_qubits + n_trash_qubits))
    ancilla_wire = n_input_qubits + n_trash_qubits
    n_wires = ancilla_wire + 1

    device = qml.device(_config.quantum.device, wires=n_wires)
    input_wires = list(range(n_input_qubits))

    @qml.qnode(device, interface="torch", diff_method=_config.quantum.diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        # Angle-encode the (classically pre-projected) embedding onto the input register.
        for i, wire in enumerate(input_wires):
            qml.RY(inputs[..., i], wires=wire)

        # Trainable encoder ansatz — the circuit being trained to compress.
        qml.StronglyEntanglingLayers(weights, wires=input_wires)

        # Reference wires are left untouched (fixed |0...0>); SWAP-test
        # them against the trash wires to score reconstruction fidelity.
        swap_test(ancilla_wire, trash_wires, reference_wires)

        # NOTE: `qml.qnn.TorchLayer` only reshapes a *list*-returning QNode's
        # batched output to (B, n_outputs); a tuple return is left in its
        # raw (n_outputs, B, 1) per-measurement shape instead. Must be a list.
        latent_expvals = [qml.expval(qml.PauliZ(w)) for w in latent_wires]
        fidelity_expval = qml.expval(qml.PauliZ(ancilla_wire))
        return [*latent_expvals, fidelity_expval]

    return circuit, n_wires


class QuantumAutoencoder(nn.Module):
    """Trainable Quantum Autoencoder that compresses `feature_dim`-D
    classical embeddings (typically 512-D ResNet18 features) into
    `n_latent_qubits` quantum-native latent dimensions directly in
    Hilbert space, via a SWAP-test reconstruction-fidelity objective — a
    trainable alternative to `pca_reducer.QuantumDimReducer`'s classical
    PCA.

    Args:
        feature_dim: Dimensionality of the input classical embedding
            (512 for the pipeline's ResNet18 features).
        n_input_qubits: Wires in the circuit's input register (after
            classical pre-projection). Must be > n_latent_qubits; default
            6 keeps total simulated wires (`n_input_qubits + n_trash + 1`)
            small enough for `default.qubit` to stay fast.
        n_latent_qubits: Wires kept as the compressed output — matches
            `config.quantum.n_qubits` by default, so the output is a
            drop-in replacement for `QuantumDimReducer`'s angle output.
        n_layers: Depth of the `StronglyEntanglingLayers` encoder ansatz.

    Forward:
        embeddings: (B, feature_dim) tensor (raw ResNet18 features).
        returns: (latent_angles, fidelity)
            latent_angles: (B, n_latent_qubits) tensor in [0, 2*pi], ready
                for a downstream angle-encoding VQC.
            fidelity: (B,) tensor in [0, 1] — SWAP-test overlap between
                the trash register and the |0...0> reference. Use
                `QuantumAutoencoder.reconstruction_loss(fidelity)` (or
                `(1 - fidelity).mean()` directly) as the QAE's training
                signal: driving fidelity -> 1 forces the trash register
                toward a fixed product state, which is what makes the
                latent register's `n_latent_qubits` sufficient to
                reconstruct the encoded information.
    """

    def __init__(
        self,
        feature_dim: int = _config.resnet.feature_dim,
        n_input_qubits: int = DEFAULT_N_INPUT_QUBITS,
        n_latent_qubits: int = N_LATENT_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
    ):
        super().__init__()
        if n_latent_qubits >= n_input_qubits:
            raise ValueError(
                f"n_latent_qubits ({n_latent_qubits}) must be < n_input_qubits ({n_input_qubits})"
            )
        self.feature_dim = feature_dim
        self.n_input_qubits = n_input_qubits
        self.n_latent_qubits = n_latent_qubits
        self.n_trash_qubits = n_input_qubits - n_latent_qubits
        self.n_layers = n_layers

        # Trainable classical pre-projection: feature_dim -> n_input_qubits,
        # so the quantum circuit only simulates a handful of wires instead
        # of one per raw ResNet feature. Trained jointly with the quantum
        # encoder against the same reconstruction-fidelity loss.
        self.classical_projection = nn.Linear(feature_dim, n_input_qubits)

        circuit, self.n_wires = build_qae_qnode(
            n_input_qubits=n_input_qubits, n_latent_qubits=n_latent_qubits, n_layers=n_layers
        )
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_input_qubits)
        self.quantum_encoder = qml.qnn.TorchLayer(circuit, weight_shapes={"weights": weight_shape})

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() != 2 or embeddings.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected input shape (B, {self.feature_dim}), got {tuple(embeddings.shape)}"
            )

        # tanh * pi keeps the projected values in [-pi, pi], a well-conditioned
        # rotation-angle range for RY encoding (avoids unbounded angles for
        # large-magnitude ResNet activations).
        projected = torch.tanh(self.classical_projection(embeddings)) * torch.pi  # (B, n_input_qubits)

        outputs = self.quantum_encoder(projected)  # (B, n_latent_qubits + 1)
        latent_expvals = outputs[..., : self.n_latent_qubits]  # (B, n_latent_qubits), in [-1, 1]
        fidelity_expval = outputs[..., self.n_latent_qubits]   # (B,), in [-1, 1]

        latent_angles = (latent_expvals + 1.0) * torch.pi   # [-1, 1] -> [0, 2*pi]
        fidelity = (fidelity_expval + 1.0) / 2.0             # [-1, 1] -> [0, 1]
        return latent_angles, fidelity

    @staticmethod
    def reconstruction_loss(fidelity: torch.Tensor) -> torch.Tensor:
        """QAE training objective: `1 - fidelity`, averaged over the batch.
        Minimizing this drives the trash register toward the |0...0>
        reference — the quantum analogue of a classical autoencoder's
        reconstruction loss, without an explicit decoder."""
        return (1.0 - fidelity).mean()

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Latent-only convenience wrapper around `forward()`, for callers
        that just want the compressed representation for downstream
        classification (e.g. `PipelineRunner.reduce_to_quantum_angles`)
        and don't need the SWAP-test fidelity — the training-time signal,
        irrelevant once the encoder is fitted and only inference is
        happening. Equivalent to `self(embeddings)[0]`.

        Args:
            embeddings: (B, feature_dim) tensor (raw ResNet18 features).

        Returns:
            (B, n_latent_qubits) tensor in [0, 2*pi], ready for a
            downstream angle-encoding VQC.
        """
        latent_angles, _fidelity = self.forward(embeddings)
        return latent_angles


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()
    torch.manual_seed(0)

    model = QuantumAutoencoder(
        feature_dim=512, n_input_qubits=DEFAULT_N_INPUT_QUBITS, n_latent_qubits=N_LATENT_QUBITS,
        n_layers=DEFAULT_N_LAYERS,
    )
    logger.info(
        "QuantumAutoencoder: %d input qubits (%d trash), %d latent qubits, %d total simulated wires",
        model.n_input_qubits, model.n_trash_qubits, model.n_latent_qubits, model.n_wires,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %d", trainable)

    # --- Forward pass smoke test ---
    batch_size = 4
    dummy_embeddings = torch.randn(batch_size, 512)
    latent_angles, fidelity = model(dummy_embeddings)

    assert latent_angles.shape == (batch_size, N_LATENT_QUBITS)
    assert fidelity.shape == (batch_size,)
    assert torch.all(latent_angles >= 0.0) and torch.all(latent_angles <= 2 * torch.pi + 1e-6)
    assert torch.all(fidelity >= 0.0) and torch.all(fidelity <= 1.0 + 1e-6)
    logger.info("Forward pass OK: embeddings %s -> latent_angles %s, fidelity %s",
                tuple(dummy_embeddings.shape), tuple(latent_angles.shape), tuple(fidelity.shape))
    logger.info("Initial mean fidelity (untrained encoder): %.4f", fidelity.mean().item())

    # --- Autograd + reconstruction-fidelity training loop integration test ---
    logger.info("Training the encoder to maximize SWAP-test fidelity...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    initial_loss: float = float("nan")
    for epoch in range(30):
        optimizer.zero_grad()
        _, fidelity = model(dummy_embeddings)
        loss = QuantumAutoencoder.reconstruction_loss(fidelity)
        loss.backward()

        if epoch == 0:
            initial_loss = loss.item()
            assert model.classical_projection.weight.grad is not None
            assert model.quantum_encoder.weights.grad is not None
            assert torch.any(model.quantum_encoder.weights.grad != 0)

        optimizer.step()
        if epoch % 5 == 0 or epoch == 29:
            logger.debug("  epoch %2d | reconstruction loss = %.4f", epoch, loss.item())

    final_loss = loss.item()
    logger.info("Initial loss: %.4f -> Final loss: %.4f", initial_loss, final_loss)
    assert final_loss < initial_loss, "Expected reconstruction loss to decrease after training"
    logger.info("Gradients flowed through the classical projection and quantum encoder; "
                "fidelity improved. All checks passed.")
