"""Checkpoint/model-health auditing, shared by every UI surface (FastAPI
`/health`, the Streamlit dashboard's sidebar) so "which model heads can
actually serve a trustworthy prediction right now" is answered once, in one
place, with one vocabulary.

Fixes AUDIT.md P1 #6 (B3/C4b): today, on this exact checkout,
`acl_checkpoint`/`meniscus_checkpoint` (see `qknee.config.config.yaml`) do
not exist on disk, and MCL has no checkpoint path configured at all — every
one of the Streamlit dashboard's three risk heads was, until this module
existed, silently scoring real ResNet18/PCA features through a randomly-
initialized quantum classifier and presenting the result with the exact
same visual weight as a genuinely trained prediction. This module is the
single source of truth `qknee/ui/dashboard.py` and `extras/api/server.py`
both call to decide whether a head may present a numeric score at all, or
must instead show "UNAVAILABLE" (see `ModelStatus`/`CheckpointInfo` below).

Two entry points, deliberately different costs:

    `quick_status(config)`      — `Path.exists()` only, no torch import, no
                                   file I/O beyond a stat call. Safe to call
                                   on every `/health` request (mirrors
                                   `extras.api.server._artifact_availability`'s
                                   existing cheap-health-probe design).
    `full_status(config)`       — actually reads and validates each
                                   checkpoint (architecture, n_qubits/
                                   n_layers, a content-hash identity).
                                   Requires torch; only call this where a
                                   torch import is already an accepted cost
                                   (a UI surface that's already loaded the
                                   full ML stack, or a test/CLI tool) — never
                                   from a fast/serverless health probe.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

ModelStatus = Literal["available", "unavailable", "unknown"]

# MCL has no `config.yaml` checkpoint path at all (unlike ACL/meniscus) —
# this project has never had a trained MCL head; it is a permanent research
# placeholder, not an accidentally-missing artifact. See `full_status`'s
# docstring and AUDIT.md B3.
MCL_PLACEHOLDER_REASON = (
    "No MCL checkpoint path is configured in config.yaml (paths.mcl_checkpoint does not exist) — "
    "MCL is a research placeholder with no trained model, not a temporarily-missing artifact."
)


@dataclass(frozen=True)
class CheckpointInfo:
    """Everything a UI needs to decide whether a head may present a
    numeric prediction, and if so, what to call it."""

    status: ModelStatus
    path: Optional[Path]
    reason: Optional[str]  # populated whenever status != "available"
    epoch: Optional[int] = None
    n_qubits: Optional[int] = None
    n_layers: Optional[int] = None
    checkpoint_id: Optional[str] = None  # short content hash — "which exact weights" identity

    @property
    def is_available(self) -> bool:
        return self.status == "available"


def _short_content_hash(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest[:12]


def inspect_vqc_checkpoint(
    path: Optional[Path],
    expected_n_qubits: int,
    expected_n_layers: int,
    *,
    unavailable_reason: Optional[str] = None,
) -> CheckpointInfo:
    """Full validation of one VQC checkpoint file: existence, that it reads
    as a dict, that it declares `n_qubits`/`n_layers` matching the target
    architecture, and a content-hash identity for the "report model
    version/checkpoint identity" requirement. Never raises — any problem
    (missing file, unreadable, architecture mismatch) becomes a
    `status="unavailable"` result with a human-readable `reason`, so a
    caller can always safely render this without a try/except of its own.

    `path=None` (no checkpoint path configured at all, e.g. MCL) or an
    explicit `unavailable_reason` short-circuits straight to
    `status="unavailable"` without attempting any file I/O.
    """
    if path is None:
        return CheckpointInfo(status="unavailable", path=None, reason=unavailable_reason or "No checkpoint path configured.")

    if not path.exists():
        return CheckpointInfo(
            status="unavailable", path=path,
            reason=unavailable_reason or f"No checkpoint found at {path}.",
        )

    try:
        import torch
    except ImportError as exc:
        return CheckpointInfo(status="unknown", path=path, reason=f"torch is not importable: {exc}")

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any read failure is a reportable "unavailable", not a crash
        return CheckpointInfo(status="unavailable", path=path, reason=f"Failed to read checkpoint: {exc}")

    if not isinstance(checkpoint, dict):
        return CheckpointInfo(
            status="unavailable", path=path, reason=f"Expected a dict checkpoint, got {type(checkpoint).__name__}.",
        )

    checkpoint_n_qubits = checkpoint.get("n_qubits")
    checkpoint_n_layers = checkpoint.get("n_layers")
    if checkpoint_n_qubits != expected_n_qubits or checkpoint_n_layers != expected_n_layers:
        return CheckpointInfo(
            status="unavailable", path=path,
            reason=(
                f"Architecture mismatch: checkpoint has n_qubits={checkpoint_n_qubits}, "
                f"n_layers={checkpoint_n_layers}, but config.quantum expects "
                f"n_qubits={expected_n_qubits}, n_layers={expected_n_layers}."
            ),
            n_qubits=checkpoint_n_qubits, n_layers=checkpoint_n_layers,
        )

    has_vqc_state = "vqc_state_dict" in checkpoint or any(
        key.startswith("vqc.") for key in checkpoint.get("model_state_dict", {})
    )
    if not has_vqc_state:
        return CheckpointInfo(
            status="unavailable", path=path,
            reason="Checkpoint has no 'vqc_state_dict' and no 'vqc.'-prefixed keys in 'model_state_dict' — "
                   "cannot recover VQC weights from it.",
            n_qubits=checkpoint_n_qubits, n_layers=checkpoint_n_layers,
        )

    return CheckpointInfo(
        status="available",
        path=path,
        reason=None,
        epoch=checkpoint.get("epoch"),
        n_qubits=checkpoint_n_qubits,
        n_layers=checkpoint_n_layers,
        checkpoint_id=_short_content_hash(path),
    )


def full_status(config) -> Dict[str, CheckpointInfo]:
    """Full, torch-validated status for every named model head this
    project has: the Streamlit dashboard's `acl`/`mcl`/`meniscus` triad,
    plus `primary` — the single unified VQC `extras/api/server.py`'s
    `/predict` actually serves (`config.paths.model_checkpoint`, falling
    back to `qknee.models.pipeline.DEFAULT_BEST_CHECKPOINT_PATH` exactly
    like `PipelineRunner.__init__` does)."""
    from qknee.models.pipeline import DEFAULT_BEST_CHECKPOINT_PATH

    n_qubits = config.quantum.n_qubits
    n_layers = config.quantum.n_layers

    primary_path = Path(config.paths.model_checkpoint)
    if not primary_path.exists():
        primary_path = DEFAULT_BEST_CHECKPOINT_PATH

    return {
        "primary": inspect_vqc_checkpoint(primary_path, n_qubits, n_layers),
        "acl": inspect_vqc_checkpoint(Path(config.paths.acl_checkpoint), n_qubits, n_layers),
        "meniscus": inspect_vqc_checkpoint(Path(config.paths.meniscus_checkpoint), n_qubits, n_layers),
        "mcl": inspect_vqc_checkpoint(None, n_qubits, n_layers, unavailable_reason=MCL_PLACEHOLDER_REASON),
    }


def status_summary(status: Dict[str, CheckpointInfo]) -> Dict[str, str]:
    """Reshapes a `full_status`/`quick_status` result into the plain
    `{"acl": "available", "mcl": "unavailable", ...}` shape a health
    endpoint or sidebar renders directly."""
    return {name: info.status for name, info in status.items()}


def quick_status(config) -> Dict[str, str]:
    """Cheap existence-only status — `Path.exists()`, no torch import, no
    file-content I/O — for a fast/serverless health probe (`GET /health`)
    that must never pay a torch import just to answer "is a checkpoint
    file present". Reports "unavailable" for MCL unconditionally (no path
    is ever configured for it), and "available"/"unavailable" for the
    others based purely on whether the file exists — NOT full architecture
    validation (see `full_status` for that). A caller that needs the
    stronger guarantee (this checkpoint will actually load and match the
    configured architecture) must use `full_status` instead."""
    from qknee.models.pipeline import DEFAULT_BEST_CHECKPOINT_PATH

    primary_path = Path(config.paths.model_checkpoint)
    if not primary_path.exists():
        primary_path = DEFAULT_BEST_CHECKPOINT_PATH

    return {
        "primary": "available" if primary_path.exists() else "unavailable",
        "acl": "available" if Path(config.paths.acl_checkpoint).exists() else "unavailable",
        "meniscus": "available" if Path(config.paths.meniscus_checkpoint).exists() else "unavailable",
        "mcl": "unavailable",
    }
