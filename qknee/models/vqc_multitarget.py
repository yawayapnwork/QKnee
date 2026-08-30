"""
Multi-target (12 RSNA Knee condition) quantum-classical heads for
`qknee.models.qknee_model.QKneeMultiTargetModel`.

Two architectural options, selected via `config.yaml`'s
`quantum.multi_target_head` ("multi_observable" | "ensemble") and built by
`build_multi_target_head`:

    1. `MultiObservableVQC` — a single 4-qubit circuit (the same angle-
       encoding + variational-block ansatz `qknee.models.vqc.VQCClassifier`
       uses) measuring 12 distinct Pauli-Z-word observables
       (`PAULI_WORD_WIRES`: Z0, Z1, Z2, Z3, Z0Z1, Z1Z2, ... — 4 single-
       qubit terms + 6 two-qubit terms + 2 three-qubit terms, all diagonal
       in the computational basis so one circuit execution measures every
       one of them), linearly projected (`nn.Linear(12, 12)`) to 12 raw
       per-condition logits.

    2. `EnsembleMultiTargetHead` — three DEDICATED, independent 4-qubit
       VQC sub-circuits (each its own trainable quantum weights and its
       own classical readout) for the primary clinical triad (ACL, MCL,
       Medial Meniscus — `RSNA_TARGET_COLUMNS[:3]`), plus one classical
       `nn.Linear(n_qubits, 9)` projecting the same 4-D angle-encoded
       input straight to the 9 secondary-finding logits — no quantum
       circuit involved for those.

Both share one interface: `(B, n_qubits) -> (B, 12)` RAW LOGITS
(pre-sigmoid) in `qknee.data.dataset.RSNA_TARGET_COLUMNS` order — never a
sigmoid-activated probability. This is deliberate: `nn.BCEWithLogitsLoss`
(used by `qknee.models.qknee_model.train_qknee_multitarget_model`) needs
raw logits for its numerically-fused sigmoid+BCE computation; calibrated
`[0.0, 1.0]` probabilities are obtained by applying `torch.sigmoid()`
*outside* the head (see `QKneeMultiTargetModel.predict_proba`).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import pennylane as qml
import torch
import torch.nn as nn

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger
from qknee.data.dataset import RSNA_TARGET_COLUMNS
from qknee.models.vqc import angle_encoding, variational_block

logger = get_logger(__name__)
_config = load_config()

N_QUBITS = _config.quantum.n_qubits
N_RSNA_TARGETS = len(RSNA_TARGET_COLUMNS)  # 12
ROTATIONS_PER_QUBIT_PER_LAYER = 3  # RX, RY, RZ — matches vqc.variational_block

MultiTargetHeadType = Literal["multi_observable", "ensemble"]

# The primary clinical triad (ensemble head) / secondary findings
# (classical projection) split — the first 3 vs. remaining 9 of
# RSNA_TARGET_COLUMNS. Asserted explicitly so a future reordering of
# RSNA_TARGET_COLUMNS fails loudly here instead of silently mislabeling
# which conditions get a dedicated quantum sub-circuit.
TRIAD_CONDITIONS: Tuple[str, str, str] = RSNA_TARGET_COLUMNS[:3]
SECONDARY_CONDITIONS: Tuple[str, ...] = RSNA_TARGET_COLUMNS[3:]
assert TRIAD_CONDITIONS == ("ACL", "MCL", "Medial Meniscus"), (
    f"Expected RSNA_TARGET_COLUMNS[:3] == ('ACL', 'MCL', 'Medial Meniscus'), got {TRIAD_CONDITIONS}"
)

# 12 distinct Pauli-Z-word observables over 4 qubits: 4 single-qubit terms,
# 6 two-qubit terms, 2 three-qubit terms. Every term is diagonal in the
# computational basis (a tensor product of PauliZ operators), so all 12
# commute and can be reported as expectation values from ONE circuit
# execution/state — no repeated state preparation per observable needed.
PAULI_WORD_WIRES: Tuple[Tuple[int, ...], ...] = (
    (0,), (1,), (2,), (3,),                              # Z0, Z1, Z2, Z3
    (0, 1), (1, 2), (2, 3), (0, 2), (0, 3), (1, 3),       # Z0Z1, Z1Z2, Z2Z3, Z0Z2, Z0Z3, Z1Z3
    (0, 1, 2), (1, 2, 3),                                  # Z0Z1Z2, Z1Z2Z3
)
assert len(PAULI_WORD_WIRES) == N_RSNA_TARGETS, (
    f"PAULI_WORD_WIRES must have exactly {N_RSNA_TARGETS} entries (one raw feature per RSNA "
    f"condition, before the linear projection), got {len(PAULI_WORD_WIRES)}"
)


def _pauli_word_observable(wires: Tuple[int, ...]):
    """Builds the `qml.PauliZ(w0) @ qml.PauliZ(w1) @ ...` operator for one
    entry of `PAULI_WORD_WIRES`."""
    observable = qml.PauliZ(wires[0])
    for wire in wires[1:]:
        observable = observable @ qml.PauliZ(wire)
    return observable


def build_multi_observable_qnode(n_qubits: int = N_QUBITS, n_layers: int = _config.quantum.n_layers):
    """Constructs the Multi-Observable VQC's QNode: angle encoding ->
    `n_layers` variational blocks (identical ansatz to
    `qknee.models.vqc.build_qnode`) -> all 12 `PAULI_WORD_WIRES`
    expectation values.

    Returns:
        A `qml.QNode` with signature `circuit(inputs, weights) ->
        List[scalar]` (12 Pauli-word expectation values, each in `[-1, 1]`).
    """
    if n_qubits != 4:
        raise ValueError(
            f"build_multi_observable_qnode's PAULI_WORD_WIRES layout is fixed for 4 qubits, got n_qubits={n_qubits}."
        )
    device = qml.device(_config.quantum.device, wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(device, interface="torch", diff_method=_config.quantum.diff_method)
    def circuit(inputs: torch.Tensor, weights: torch.Tensor):
        angle_encoding(inputs, wires)
        for layer in range(n_layers):
            variational_block(weights[layer], wires)
        return [qml.expval(_pauli_word_observable(word_wires)) for word_wires in PAULI_WORD_WIRES]

    return circuit


class MultiObservableVQC(nn.Module):
    """Multi-Observable VQC: a single trainable 4-qubit circuit measuring
    12 distinct Pauli-Z-word observables (`PAULI_WORD_WIRES`, each in
    `[-1, 1]`), linearly projected to 12 raw per-condition logits.

    Args:
        n_qubits: Must be 4 — `PAULI_WORD_WIRES`' layout is fixed for a
            4-qubit register.
        n_layers: Depth of the variational-block ansatz.

    Forward:
        x: `(B, n_qubits)` tensor, values in `[0, 2*pi]`.
        returns: `(B, 12)` RAW LOGITS (pre-sigmoid) — see module docstring.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = _config.quantum.n_layers):
        super().__init__()
        if n_qubits != 4:
            raise ValueError(f"MultiObservableVQC requires n_qubits=4 (fixed 12-observable layout), got {n_qubits}")
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        circuit = build_multi_observable_qnode(n_qubits=n_qubits, n_layers=n_layers)
        weight_shapes = {"weights": (n_layers, n_qubits, ROTATIONS_PER_QUBIT_PER_LAYER)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # The "linear projection layer": 12 raw Pauli-word expectation
        # values -> 12 raw per-condition logits.
        self.projection = nn.Linear(len(PAULI_WORD_WIRES), N_RSNA_TARGETS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != self.n_qubits:
            raise ValueError(f"Expected input shape (B, {self.n_qubits}), got {tuple(x.shape)}")
        observable_expvals = self.quantum_layer(x)  # (B, 12), each in [-1, 1]
        return self.projection(observable_expvals)   # (B, 12) raw logits


class EnsembleMultiTargetHead(nn.Module):
    """Ensemble / Multi-Circuit head: one DEDICATED 4-qubit VQC
    sub-circuit per primary-triad condition (`TRIAD_CONDITIONS` — ACL,
    MCL, Medial Meniscus), each with its own independent trainable
    quantum weights and its own `Linear(n_qubits, 1)` readout, plus a
    single classical `Linear(n_qubits, 9)` projecting the same 4-D
    angle-encoded input directly to the `SECONDARY_CONDITIONS` logits —
    no quantum circuit involved for the secondary findings.

    Args:
        n_qubits: Number of qubits per triad sub-circuit.
        n_layers: Variational depth of each triad sub-circuit.

    Forward:
        x: `(B, n_qubits)` tensor, values in `[0, 2*pi]`.
        returns: `(B, 12)` RAW LOGITS, in `RSNA_TARGET_COLUMNS` order
            (triad logits first, then secondary) — see module docstring.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = _config.quantum.n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Local import: qknee.models.vqc has no dependency on this module,
        # but importing it at call time (rather than module top-level)
        # keeps the dependency direction explicit at the one place it's
        # actually needed.
        from qknee.models.vqc import VQCClassifier

        # Each triad circuit is a full, independent VQCClassifier — its
        # own quantum weights and its own Linear(n_qubits, 1) readout —
        # i.e. genuinely "dedicated sub-circuits", not a shared one.
        self.triad_circuits = nn.ModuleList([
            VQCClassifier(n_qubits=n_qubits, n_layers=n_layers) for _ in TRIAD_CONDITIONS
        ])
        self.secondary_projection = nn.Linear(n_qubits, len(SECONDARY_CONDITIONS))

    @staticmethod
    def _triad_logit(circuit: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Reproduces one triad `VQCClassifier`'s quantum-layer + readout
        computation, but stops *before* its internal `Sigmoid` — that
        module is built for single-target `[0, 1]` probability output, so
        pulling the pre-sigmoid logit back out keeps this ensemble head in
        logit-space end-to-end for `BCEWithLogitsLoss`."""
        expvals = circuit.quantum_layer(x)  # (B, n_qubits)
        return circuit.readout(expvals)     # (B, 1), raw logit (pre-sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != self.n_qubits:
            raise ValueError(f"Expected input shape (B, {self.n_qubits}), got {tuple(x.shape)}")

        triad_logits = torch.cat(
            [self._triad_logit(circuit, x) for circuit in self.triad_circuits], dim=1,
        )  # (B, 3)
        secondary_logits = self.secondary_projection(x)  # (B, 9)
        return torch.cat([triad_logits, secondary_logits], dim=1)  # (B, 12)


def build_multi_target_head(
    head_type: Optional[MultiTargetHeadType] = None,
    n_qubits: int = N_QUBITS,
    n_layers: int = _config.quantum.n_layers,
) -> nn.Module:
    """Factory selecting the multi-target head architecture — the
    `config.yaml`'s `quantum.multi_target_head` setting turned into an
    actual module, so callers (`QKneeMultiTargetModel`) don't need an
    `if/elif` of their own.

    Args:
        head_type: `"multi_observable"` or `"ensemble"`; defaults to
            `config.quantum.multi_target_head`.
        n_qubits: Qubits per circuit.
        n_layers: Variational depth per circuit.

    Returns:
        A `(B, n_qubits) -> (B, 12)` raw-logits `nn.Module`.

    Raises:
        ValueError: for an unrecognized `head_type`.
    """
    resolved_head_type = head_type or _config.quantum.multi_target_head
    if resolved_head_type == "multi_observable":
        return MultiObservableVQC(n_qubits=n_qubits, n_layers=n_layers)
    elif resolved_head_type == "ensemble":
        return EnsembleMultiTargetHead(n_qubits=n_qubits, n_layers=n_layers)
    raise ValueError(
        f"Unknown multi_target_head '{resolved_head_type}'; expected 'multi_observable' or 'ensemble'."
    )


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()
    torch.manual_seed(0)

    dummy_input = torch.rand(5, N_QUBITS) * 2 * torch.pi

    for head_type in ("multi_observable", "ensemble"):
        logger.info("=== %s ===", head_type)
        head = build_multi_target_head(head_type)
        trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %d", trainable)

        logits = head(dummy_input)
        assert logits.shape == (5, N_RSNA_TARGETS), f"Unexpected shape {tuple(logits.shape)}"
        logger.info("Logits shape: %s (raw, pre-sigmoid)", tuple(logits.shape))

        probabilities = torch.sigmoid(logits)
        assert torch.all(probabilities >= 0.0) and torch.all(probabilities <= 1.0)
        logger.info("Sigmoid probabilities in [0, 1]: OK")

        # Gradient flow through every trainable parameter.
        logits.sum().backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"{head_type}: no gradient reached parameter '{name}'"
            assert torch.isfinite(param.grad).all()
        logger.info("Gradient flow through every trainable parameter: OK")

    logger.info("All vqc_multitarget.py checks passed.")
