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
