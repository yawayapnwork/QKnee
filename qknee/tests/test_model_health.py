"""
Tests for `qknee.observability.model_health` — the checkpoint-auditing
module AUDIT.md P1 #6 introduces so no UI surface can silently present a
randomly-initialized model's output as a trustworthy prediction.

Covers:
    1. `inspect_vqc_checkpoint`: missing file, unreadable file, architecture
       (n_qubits/n_layers) mismatch, and a genuinely valid checkpoint —
       including its "checkpoint identity" (content hash) and epoch.
    2. `quick_status`/`full_status`/`status_summary` against a real
       (monkeypatched) config, including MCL's permanent "unavailable"
       placeholder status.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qknee.observability.model_health import (
    MCL_PLACEHOLDER_REASON,
    CheckpointInfo,
    full_status,
    inspect_vqc_checkpoint,
    quick_status,
    status_summary,
)


def _write_checkpoint(path: Path, *, n_qubits: int = 4, n_layers: int = 3, epoch=5, valid_keys=True) -> None:
    checkpoint = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "epoch": epoch,
    }
    if valid_keys:
        checkpoint["vqc_state_dict"] = {"readout.weight": torch.zeros(1, n_qubits), "readout.bias": torch.zeros(1)}
    torch.save(checkpoint, path)


class TestInspectVqcCheckpoint:
    def test_missing_path_argument_is_unavailable_without_touching_disk(self):
        info = inspect_vqc_checkpoint(None, 4, 3, unavailable_reason="custom reason")
        assert info.status == "unavailable"
        assert info.reason == "custom reason"
        assert info.path is None

    def test_nonexistent_file_is_unavailable(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.pt"
        info = inspect_vqc_checkpoint(missing, 4, 3)
        assert info.status == "unavailable"
        assert "No checkpoint found" in info.reason
        assert info.path == missing

    def test_unreadable_file_is_unavailable_not_a_crash(self, tmp_path: Path):
        garbage = tmp_path / "garbage.pt"
        garbage.write_bytes(b"not a real torch checkpoint")
        info = inspect_vqc_checkpoint(garbage, 4, 3)
        assert info.status == "unavailable"
        assert "Failed to read checkpoint" in info.reason

    def test_non_dict_checkpoint_is_unavailable(self, tmp_path: Path):
        path = tmp_path / "tensor_only.pt"
        torch.save(torch.zeros(4), path)
        info = inspect_vqc_checkpoint(path, 4, 3)
        assert info.status == "unavailable"
        assert "Expected a dict" in info.reason

    def test_architecture_mismatch_is_unavailable(self, tmp_path: Path):
        path = tmp_path / "wrong_arch.pt"
        _write_checkpoint(path, n_qubits=8, n_layers=3)
        info = inspect_vqc_checkpoint(path, expected_n_qubits=4, expected_n_layers=3)
        assert info.status == "unavailable"
        assert "Architecture mismatch" in info.reason
        assert info.n_qubits == 8

    def test_missing_vqc_state_is_unavailable(self, tmp_path: Path):
        path = tmp_path / "no_state.pt"
        _write_checkpoint(path, valid_keys=False)
        info = inspect_vqc_checkpoint(path, 4, 3)
        assert info.status == "unavailable"
        assert "no 'vqc_state_dict'" in info.reason

    def test_valid_checkpoint_is_available_with_identity(self, tmp_path: Path):
        path = tmp_path / "trained.pt"
        _write_checkpoint(path, n_qubits=4, n_layers=3, epoch=7)

        info = inspect_vqc_checkpoint(path, expected_n_qubits=4, expected_n_layers=3)

        assert info.status == "available"
        assert info.reason is None
        assert info.epoch == 7
        assert info.n_qubits == 4
        assert info.n_layers == 3
        assert info.checkpoint_id is not None
        assert len(info.checkpoint_id) == 12

    def test_checkpoint_identity_changes_when_file_content_changes(self, tmp_path: Path):
        path = tmp_path / "v1.pt"
        _write_checkpoint(path, epoch=1)
        info_v1 = inspect_vqc_checkpoint(path, 4, 3)

        _write_checkpoint(path, epoch=2)
        info_v2 = inspect_vqc_checkpoint(path, 4, 3)

        assert info_v1.checkpoint_id != info_v2.checkpoint_id


def _fake_config(tmp_path: Path, *, acl_exists: bool, meniscus_exists: bool, primary_exists: bool):
    acl_path = tmp_path / "acl_vqc.pt"
    meniscus_path = tmp_path / "meniscus_vqc.pt"
    primary_path = tmp_path / "qknee_model.pt"
    for path, exists in ((acl_path, acl_exists), (meniscus_path, meniscus_exists), (primary_path, primary_exists)):
        if exists:
            _write_checkpoint(path)

    return SimpleNamespace(
        paths=SimpleNamespace(
            acl_checkpoint=acl_path, meniscus_checkpoint=meniscus_path, model_checkpoint=primary_path,
        ),
        quantum=SimpleNamespace(n_qubits=4, n_layers=3),
    )


@pytest.fixture(autouse=True)
def _isolate_default_best_checkpoint_fallback(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """`full_status`/`quick_status` mirror `PipelineRunner.__init__`'s real
    fallback to `DEFAULT_BEST_CHECKPOINT_PATH` when the configured
    `model_checkpoint` is missing — correct production behavior, but that
    path is a real, repo-relative file that may genuinely exist on this
    checkout (see `qknee/artifacts/checkpoints/best_checkpoint.pt`).
    Pointed at a guaranteed-nonexistent path so every test in this module
    is isolated from whatever training artifacts happen to be on disk."""
    import qknee.models.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "DEFAULT_BEST_CHECKPOINT_PATH", tmp_path_factory.mktemp("no-fallback") / "unused.pt",
    )


class TestFullStatus:
    def test_mcl_is_always_unavailable_with_the_placeholder_reason(self, tmp_path: Path):
        config = _fake_config(tmp_path, acl_exists=True, meniscus_exists=True, primary_exists=True)
        status = full_status(config)
        assert status["mcl"].status == "unavailable"
        assert status["mcl"].reason == MCL_PLACEHOLDER_REASON

    def test_reports_available_only_for_heads_with_a_real_checkpoint_on_disk(self, tmp_path: Path):
        config = _fake_config(tmp_path, acl_exists=True, meniscus_exists=False, primary_exists=True)
        status = full_status(config)

        assert status["primary"].status == "available"
        assert status["acl"].status == "available"
        assert status["meniscus"].status == "unavailable"
        assert status["mcl"].status == "unavailable"

    def test_status_summary_reshapes_to_plain_strings(self, tmp_path: Path):
        config = _fake_config(tmp_path, acl_exists=True, meniscus_exists=True, primary_exists=True)
        summary = status_summary(full_status(config))

        assert summary == {"primary": "available", "acl": "available", "meniscus": "available", "mcl": "unavailable"}


class TestQuickStatus:
    def test_matches_full_status_availability_for_a_valid_checkpoint(self, tmp_path: Path):
        config = _fake_config(tmp_path, acl_exists=True, meniscus_exists=False, primary_exists=True)

        quick = quick_status(config)
        full = status_summary(full_status(config))

        assert quick == full  # both agree when the on-disk file is genuinely a valid checkpoint

    def test_never_imports_torch_for_a_missing_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """`quick_status` must be answerable from Path.exists() alone --
        this is what makes it safe to call from a fast health probe."""
        config = _fake_config(tmp_path, acl_exists=False, meniscus_exists=False, primary_exists=False)
        summary = quick_status(config)
        assert summary == {"primary": "unavailable", "acl": "unavailable", "meniscus": "unavailable", "mcl": "unavailable"}
