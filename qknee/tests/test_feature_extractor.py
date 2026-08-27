"""
Unit tests: ResNet18 feature extractor output shape, and PCA/angle-scaling
output bounds.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qknee.models.pca_reducer import N_QUANTUM_DIMS, QuantumDimReducer


class TestResNetFeatureExtractorShape:
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_slice_output_shape_is_batch_by_512(self, resnet_extractor, batch_size):
        x = torch.rand(batch_size, 3, 224, 224)
        with torch.no_grad():
            features = resnet_extractor(x)

        assert features.shape == (batch_size, 512)

    def test_slice_output_is_finite(self, resnet_extractor, dummy_image_batch):
        with torch.no_grad():
            features = resnet_extractor(dummy_image_batch)

        assert torch.isfinite(features).all()

    def test_volume_output_shape_is_batch_by_512(self, resnet_extractor):
        batch_size, n_slices = 2, 5
        x = torch.rand(batch_size, n_slices, 3, 224, 224)
        with torch.no_grad():
            features = resnet_extractor(x)

        assert features.shape == (batch_size, 512)

    def test_rejects_wrong_channel_count(self, resnet_extractor):
        x = torch.rand(2, 1, 224, 224)  # single-channel, not 3
        with pytest.raises(ValueError):
            resnet_extractor(x)


class TestPCAAngleScalingBounds:
    def test_output_shape_is_n_by_4(self, fitted_reducer, dummy_resnet_features):
        angles = fitted_reducer.transform(dummy_resnet_features)
        assert angles.shape == (dummy_resnet_features.shape[0], N_QUANTUM_DIMS)

    def test_output_bounded_in_zero_to_two_pi(self, fitted_reducer, dummy_resnet_features):
        angles = fitted_reducer.transform(dummy_resnet_features)

        assert np.all(angles >= 0.0)
        assert np.all(angles <= 2 * np.pi)

    def test_bounds_hold_on_out_of_distribution_inputs(self, fitted_reducer):
        """Inputs far outside the training distribution must still clip into
        [0, 2*pi] rather than producing out-of-range angles for the VQC."""
        rng = np.random.default_rng(99)
        extreme_features = rng.normal(loc=0, scale=50, size=(10, 512)).astype(np.float32)

        angles = fitted_reducer.transform(extreme_features)

        assert np.all(angles >= 0.0)
        assert np.all(angles <= 2 * np.pi)

    def test_transform_before_fit_raises(self):
        unfitted = QuantumDimReducer()
        with pytest.raises(RuntimeError):
            unfitted.transform(np.zeros((1, 512), dtype=np.float32))

    def test_explained_variance_has_four_components_and_sums_leq_one(self, fitted_reducer):
        evr = fitted_reducer.explained_variance_ratio_
        assert evr.shape == (N_QUANTUM_DIMS,)
        assert np.all(evr >= 0.0)
        assert evr.sum() <= 1.0 + 1e-8
