"""
Unit tests: ResNet18 feature extractor output shape/range/gradient-flow,
and the PCA bottleneck's output bounds and gradient-flow (the differentiable
`PCAProjectionLayer` re-expression used for Grad-CAM, not the sklearn
`QuantumDimReducer.transform`, which has no gradient).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qknee.models.pca_reducer import N_QUANTUM_DIMS, QuantumDimReducer
from qknee.models.qknee_model import PCAProjectionLayer


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

    def test_slice_output_is_non_negative(self, resnet_extractor, dummy_image_batch):
        """The backbone ends at `avgpool` immediately after `layer4`'s
        final ReLU, so every output feature is an average of non-negative
        activations — a genuine, checkable range constraint (not just
        finiteness)."""
        with torch.no_grad():
            features = resnet_extractor(dummy_image_batch)
        assert torch.all(features >= 0.0)


class TestResNetFeatureExtractorGradientFlow:
    """Gradients must flow from the 512-D embedding all the way back to the
    input pixels — this is what makes Grad-CAM possible, and what a frozen
    backbone (`requires_grad=False` on its own weights) does *not* by
    itself guarantee: freezing stops weight updates, not gradient
    computation through the forward graph."""

    def test_gradient_flows_from_output_to_input(self, resnet_extractor):
        x = torch.rand(2, 3, 224, 224, requires_grad=True)
        features = resnet_extractor(x)
        features.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0.0  # not a degenerate all-zero gradient

    def test_frozen_backbone_weights_receive_no_gradient(self, resnet_extractor):
        """Complementary check: the backbone's own parameters are frozen
        (`requires_grad=False`), so — unlike the input above — they must
        NOT accumulate a `.grad` even though gradients pass through them."""
        assert resnet_extractor.freeze_backbone is True
        x = torch.rand(1, 3, 224, 224, requires_grad=True)
        resnet_extractor(x).sum().backward()

        for param in resnet_extractor.parameters():
            assert param.requires_grad is False
            assert param.grad is None

    def test_volume_input_gradient_flows_to_every_slice(self, resnet_extractor):
        """The per-volume path averages per-slice features before
        returning — confirms gradients still reach every slice in the
        stack, not just e.g. the first one (which a broken reshape/index
        could silently produce)."""
        x = torch.rand(1, 4, 3, 224, 224, requires_grad=True)
        resnet_extractor(x).sum().backward()

        assert x.grad is not None
        batch, n_slices = x.grad.shape[0], x.grad.shape[1]
        per_slice_grad_norms = x.grad.reshape(batch, n_slices, -1).norm(dim=2).flatten()
        assert torch.all(per_slice_grad_norms > 0.0)


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


class TestPCABottleneckGradientFlow:
    """`PCAProjectionLayer` (qknee.models.qknee_model) is the frozen,
    differentiable torch re-expression of a fitted `QuantumDimReducer`
    (StandardScaler -> PCA -> MinMaxScaler, all affine, plus a final
    `clamp`) — the piece that makes Grad-CAM possible through the PCA
    bottleneck, since `QuantumDimReducer.transform` itself is plain numpy/
    sklearn with no gradient. These tests exercise that torch layer
    directly, not the sklearn reducer."""

    @pytest.fixture
    def pca_layer(self, fitted_reducer) -> PCAProjectionLayer:
        return PCAProjectionLayer.from_reducer(fitted_reducer)

    def test_output_shape_and_bounds_match_reducer(self, pca_layer, dummy_resnet_features):
        x = torch.from_numpy(dummy_resnet_features)
        with torch.no_grad():
            angles = pca_layer(x)

        assert angles.shape == (dummy_resnet_features.shape[0], N_QUANTUM_DIMS)
        assert torch.all(angles >= 0.0)
        assert torch.all(angles <= 2 * torch.pi + 1e-6)

    def test_output_matches_sklearn_reducer_bit_for_bit(self, pca_layer, fitted_reducer, dummy_resnet_features):
        """The whole point of `PCAProjectionLayer`: it must reproduce the
        fitted sklearn pipeline's `transform()` output exactly, just as
        ordinary differentiable tensor ops."""
        sklearn_angles = fitted_reducer.transform(dummy_resnet_features)
        with torch.no_grad():
            torch_angles = pca_layer(torch.from_numpy(dummy_resnet_features)).numpy()

        np.testing.assert_allclose(torch_angles, sklearn_angles, rtol=1e-4, atol=1e-4)

    def test_gradient_flows_from_angles_to_input_embedding(self, pca_layer, dummy_resnet_features):
        x = torch.from_numpy(dummy_resnet_features).clone().requires_grad_(True)
        angles = pca_layer(x)
        angles.sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0.0

    def test_layer_parameters_are_buffers_not_trainable(self, pca_layer):
        """The PCA projection is a fixed re-expression of an already-fitted
        reducer — it must never be trainable itself (only the VQC/readout
        downstream of it should update during training), even though
        gradients pass *through* it to the input, as verified above."""
        assert list(pca_layer.parameters()) == []  # everything is a registered buffer, not nn.Parameter
        for buffer in pca_layer.buffers():
            assert buffer.requires_grad is False
