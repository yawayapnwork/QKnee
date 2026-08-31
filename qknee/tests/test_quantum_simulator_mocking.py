"""
Tests that mock the PennyLane quantum simulator layer, so pipeline behavior
can be exercised:
    - independently of the real (slower) `default.qubit` simulation, and
    - under conditions that are hard to trigger with a real simulator on
      demand, such as the backend refusing work because a NISQ resource
      limit (qubit count / shot budget / queue depth) has been reached.

Mocking is done via `unittest.mock.patch.object` on the `forward` method of
`VQCClassifier.quantum_layer` (the TorchLayer instance itself is a
registered `nn.Module` submodule, so it must be patched at the method
level rather than reassigned wholesale), so these tests never touch the
real PennyLane device.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from qknee.models.qknee_model import QKneeModel
from qknee.models.vqc import VQCClassifier


class SimulatorResourceLimitError(RuntimeError):
    """Stand-in for the error a real NISQ backend/cloud simulator would
    raise when a resource limit (max qubits, shot budget, queued-job cap)
    is exceeded."""


class TestMockedQuantumLayerOutput:
    """Replaces the quantum layer's output with controlled values so the
    classical readout (Linear + Sigmoid) can be tested in isolation, fast
    and deterministically, without running the real circuit simulation."""

    def test_classical_readout_handles_extreme_expvals(self):
        model = VQCClassifier(n_qubits=4, n_layers=3)
        batch_size = 3
        # PauliZ expectation values are always in [-1, 1]; exercise both
        # boundary values the simulator could legitimately return.
        extreme_expvals = torch.tensor(
            [[1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]
        )

        with patch.object(model.quantum_layer, "forward", return_value=extreme_expvals):
            output = model(torch.rand(batch_size, 4))

        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
        assert torch.isfinite(output).all()

    def test_end_to_end_pipeline_with_mocked_quantum_layer(self, qknee_model, dummy_image_batch):
        """Full QKneeModel forward pass, but with the quantum simulation
        step swapped for a fixed mock — verifies ResNet -> PCA -> (mocked)
        VQC -> Sigmoid wiring end-to-end without depending on PennyLane's
        actual circuit execution."""
        batch_size = dummy_image_batch.shape[0]
        mock_expvals = torch.zeros(batch_size, 4)

        with patch.object(qknee_model.vqc.quantum_layer, "forward", return_value=mock_expvals):
            output = qknee_model(dummy_image_batch)

        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)


class TestSimulatorResourceLimitBehavior:
    """Simulates the quantum backend refusing to execute because a NISQ
    resource limit was reached, and verifies the pipeline fails loudly and
    predictably (a clear exception) rather than silently returning a
    corrupted or default prediction."""

    def test_quantum_layer_failure_propagates_as_clear_error(self):
        model = VQCClassifier(n_qubits=4, n_layers=3)

        with patch.object(
            model.quantum_layer,
            "forward",
            side_effect=SimulatorResourceLimitError(
                "default.qubit: requested circuit exceeds the configured qubit/shot budget"
            ),
        ):
            with pytest.raises(SimulatorResourceLimitError, match="qubit/shot budget"):
                model(torch.rand(2, 4))

    def test_full_pipeline_failure_propagates_from_quantum_stage(self, qknee_model, dummy_image_batch):
        with patch.object(
            qknee_model.vqc.quantum_layer,
            "forward",
            side_effect=SimulatorResourceLimitError("simulator queue depth limit reached"),
        ):
            with pytest.raises(SimulatorResourceLimitError):
                qknee_model(dummy_image_batch)

    def test_resnet_and_pca_stages_still_run_before_failure(self, qknee_model, dummy_image_batch):
        """Confirms the failure genuinely originates at the quantum stage —
        i.e. ResNet18 + PCA (the classically-cheap, always-available part
        of the pipeline) still execute even when the quantum backend is
        unavailable, which matters for any future fallback/triage logic
        (e.g. falling back to the classical ResNet-only baseline from
        evaluate.py when the NISQ simulator is over capacity)."""
        resnet_call_count = 0
        original_resnet_forward = qknee_model.resnet.forward

        def _counting_resnet_forward(*args, **kwargs):
            nonlocal resnet_call_count
            resnet_call_count += 1
            return original_resnet_forward(*args, **kwargs)

        with patch.object(qknee_model.resnet, "forward", side_effect=_counting_resnet_forward), patch.object(
            qknee_model.vqc.quantum_layer,
            "forward",
            side_effect=SimulatorResourceLimitError("simulator unavailable"),
        ):
            with pytest.raises(SimulatorResourceLimitError):
                qknee_model(dummy_image_batch)

        assert resnet_call_count == 1


class TestQuantumDeviceSafetyGuard:
    """`qknee.models.vqc.load_quantum_device` is the single shared guard
    every VQC ansatz's `build_qnode`/circuit-builder routes through —
    verifies it actually falls back to `default.qubit` (never crashes the
    caller) when the configured device/plugin is unavailable, memory-
    exhausted, or otherwise broken, and that every VQC module in this
    project (`vqc.py`, `vqc_data_reuploading.py`, `vqc_multitarget.py`,
    `vqc_strongly_entangling.py`) actually uses it rather than calling
    `qml.device(...)` directly and unguarded."""

    def test_default_qubit_loads_normally(self):
        from qknee.models.vqc import load_quantum_device

        device = load_quantum_device("default.qubit", 4)
        assert device is not None

    def test_falls_back_to_default_qubit_for_an_unknown_device_name(self, caplog: pytest.LogCaptureFixture):
        from qknee.models.vqc import load_quantum_device

        with caplog.at_level("WARNING", logger="qknee.models.vqc"):
            device = load_quantum_device("this.device.does.not.exist", 4)

        assert device is not None
        assert device.name == "default.qubit"
        assert any("falling back to" in record.message.lower() for record in caplog.records)

    def test_falls_back_when_the_configured_device_raises_a_memory_error(self):
        """Simulates an accelerator backend failing to allocate its
        state-vector buffer — `MemoryError` is a plain `Exception`
        subclass, so the guard's generic `except Exception` already
        covers it; this pins that behavior explicitly."""
        from unittest.mock import patch

        from qknee.models import vqc as vqc_module

        real_device = vqc_module.qml.device

        def _raise_memory_error(device_name, **kwargs):
            if device_name == "some.accelerator.backend":
                raise MemoryError("failed to allocate state-vector buffer")
            return real_device(device_name, **kwargs)

        with patch.object(vqc_module.qml, "device", side_effect=_raise_memory_error):
            device = vqc_module.load_quantum_device("some.accelerator.backend", 4)

        assert device is not None

    def test_reraises_if_default_qubit_itself_is_broken(self):
        """No infinite fallback loop: if even `default.qubit` fails to
        load, the guard must propagate that failure rather than retry
        forever or silently return `None`."""
        from unittest.mock import patch

        from qknee.models import vqc as vqc_module

        with patch.object(
            vqc_module.qml, "device", side_effect=RuntimeError("simulated total device failure"),
        ):
            with pytest.raises(RuntimeError, match="simulated total device failure"):
                vqc_module.load_quantum_device("default.qubit", 4)

    @pytest.mark.parametrize(
        "module_name, builder_name, builder_kwargs",
        [
            ("qknee.models.vqc", "build_qnode", {"n_qubits": 4, "n_layers": 1}),
            ("qknee.models.vqc_data_reuploading", "build_qnode", {"n_qubits": 4, "n_layers": 1}),
            ("qknee.models.vqc_multitarget", "build_multi_observable_qnode", {"n_qubits": 4, "n_layers": 1}),
            ("qknee.models.vqc_strongly_entangling", "build_qnode", {"n_qubits": 4, "n_layers": 1}),
        ],
    )
    def test_every_vqc_ansatz_builds_successfully_when_the_configured_device_is_broken(
        self, module_name, builder_name, builder_kwargs, monkeypatch: pytest.MonkeyPatch,
    ):
        """The real regression test for this hardening pass: patches each
        module's own `_config.quantum.device` to a nonexistent backend
        name and confirms `build_qnode`/`build_multi_observable_qnode`
        still succeeds (falls back to `default.qubit` via
        `load_quantum_device`), rather than raising straight out of a
        direct, unguarded `qml.device(...)` call — the exact gap
        `vqc_strongly_entangling.py` had before this hardening pass."""
        import dataclasses
        import importlib

        module = importlib.import_module(module_name)
        # `QuantumConfig`/`QKneeConfig` are frozen dataclasses (see
        # qknee.config.loader) — can't assign `.device` in place, so swap
        # in a fresh config via `dataclasses.replace` instead.
        broken_quantum_config = dataclasses.replace(
            module._config.quantum, device="totally.nonexistent.accelerator.plugin",
        )
        broken_config = dataclasses.replace(module._config, quantum=broken_quantum_config)
        monkeypatch.setattr(module, "_config", broken_config)

        builder = getattr(module, builder_name)
        circuit = builder(**builder_kwargs)

        assert circuit is not None
