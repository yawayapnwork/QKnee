"""
Determinism tests: given a fixed random seed, the model's initialization
and predictions must be exactly reproducible. This matters clinically —
the same slice must not yield a different risk score on repeat runs, and
two freshly-initialized models built with the same seed must be identical
(so training runs are comparable/reproducible across environments).
"""

from __future__ import annotations

import torch

from qknee.models.qknee_model import QKneeModel
from qknee.models.vqc import VQCClassifier

pytestmark = []


class TestForwardPassDeterminism:
    def test_repeated_forward_pass_is_identical(self, qknee_model, dummy_image_batch):
        qknee_model.eval()
        with torch.no_grad():
            output_a = qknee_model(dummy_image_batch)
            output_b = qknee_model(dummy_image_batch)

        torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)

    def test_same_seed_same_input_reproducible_across_fresh_instances(self, fitted_reducer):
        """Two independently constructed models, both seeded identically
        before construction, must produce bit-identical initial weights and
        predictions."""
        image = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))

        torch.manual_seed(1234)
        model_a = QKneeModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=3)
        model_a.eval()

        torch.manual_seed(1234)
        model_b = QKneeModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=3)
        model_b.eval()

        with torch.no_grad():
            output_a = model_a(image)
            output_b = model_b(image)

        torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)

    def test_different_seeds_diverge(self, fitted_reducer):
        """Sanity check for the determinism tests themselves: different
        seeds must NOT coincidentally produce identical VQC weights (which
        would make the "same seed" tests above vacuously true)."""
        torch.manual_seed(1)
        model_a = VQCClassifier(n_qubits=4, n_layers=3)

        torch.manual_seed(2)
        model_b = VQCClassifier(n_qubits=4, n_layers=3)

        weights_equal = torch.equal(
            model_a.quantum_layer.weights, model_b.quantum_layer.weights
        )
        assert not weights_equal


class TestTrainingDeterminism:
    def test_training_loop_is_reproducible_with_fixed_seed(self, fitted_reducer):
        from qknee.models.qknee_model import train_qknee_model

        image = torch.rand(6, 3, 224, 224, generator=torch.Generator().manual_seed(0))
        labels = torch.tensor([0, 1, 0, 1, 1, 0])

        def _train_fresh_model() -> list[float]:
            torch.manual_seed(2024)
            model = QKneeModel(pca_reducer=fitted_reducer, n_qubits=4, n_layers=3)
            return train_qknee_model(model, image, labels, n_epochs=5, lr=0.05, log_every=100)

        history_a = _train_fresh_model()
        history_b = _train_fresh_model()

        assert history_a == history_b
