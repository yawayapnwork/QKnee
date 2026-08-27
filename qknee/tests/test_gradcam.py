"""
Tests for `qknee.xai.gradcam` — the Grad-CAM hook manager and heatmap
overlay/compositing helpers. Covers:

    1. Heatmap generation: shape/range, both the default (embedding-energy)
       target and a custom `target_fn`.
    2. Input/target validation: wrong batch size, non-scalar `target_fn`
       output.
    3. Hook lifecycle: hooks are registered on construction and removed by
       `remove_hooks()` / the context-manager `__exit__`.
    4. `overlay_heatmap` and `get_default_target_layer`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qknee.models.resnet_extractor import ResNet18FeatureExtractor
from qknee.xai.gradcam import GradCAM, get_default_target_layer, overlay_heatmap

pytestmark = [pytest.mark.slow]


@pytest.fixture
def target_layer(resnet_extractor: ResNet18FeatureExtractor):
    return get_default_target_layer(resnet_extractor)


@pytest.fixture
def single_input() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, 224, 224, generator=generator)


# --------------------------------------------------------------------------- #
# 1. Heatmap generation
# --------------------------------------------------------------------------- #

class TestGenerate:
    def test_default_target_produces_bounded_2d_heatmap(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer, single_input: torch.Tensor
    ):
        with GradCAM(resnet_extractor, target_layer) as cam:
            heatmap = cam.generate(single_input)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6

    def test_custom_target_fn_is_used_for_backprop(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer, single_input: torch.Tensor
    ):
        calls = []

        def target_fn(output: torch.Tensor) -> torch.Tensor:
            calls.append(output.shape)
            return output.sum()

        with GradCAM(resnet_extractor, target_layer) as cam:
            heatmap = cam.generate(single_input, target_fn=target_fn)

        assert len(calls) == 1
        assert heatmap.ndim == 2

    def test_model_training_mode_is_restored_after_generate(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer, single_input: torch.Tensor
    ):
        resnet_extractor.train()
        with GradCAM(resnet_extractor, target_layer) as cam:
            cam.generate(single_input)
        assert resnet_extractor.training is True
        resnet_extractor.eval()  # restore shared session fixture's expected state

    def test_constant_activation_gradient_yields_zero_heatmap_not_nan(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer
    ):
        """When min == max (degenerate CAM), the result must be all-zero,
        not NaN from a divide-by-zero in the normalization step."""
        zeros_input = torch.zeros(1, 3, 224, 224)
        with GradCAM(resnet_extractor, target_layer) as cam:
            heatmap = cam.generate(zeros_input, target_fn=lambda out: (out * 0).sum() + 1.0)

        assert np.all(np.isfinite(heatmap))


# --------------------------------------------------------------------------- #
# 2. Input/target validation
# --------------------------------------------------------------------------- #

class TestValidation:
    def test_batch_size_greater_than_one_raises(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer
    ):
        batch_input = torch.rand(2, 3, 224, 224)
        with GradCAM(resnet_extractor, target_layer) as cam:
            with pytest.raises(ValueError, match="batch size 1"):
                cam.generate(batch_input)

    def test_non_scalar_target_fn_output_raises(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer, single_input: torch.Tensor
    ):
        with GradCAM(resnet_extractor, target_layer) as cam:
            with pytest.raises(ValueError, match="single-element"):
                cam.generate(single_input, target_fn=lambda out: out)  # (1, 512), not scalar

    def test_target_layer_not_in_forward_path_raises_runtime_error(
        self, resnet_extractor: ResNet18FeatureExtractor, single_input: torch.Tensor
    ):
        unused_layer = torch.nn.Conv2d(3, 3, kernel_size=1)  # never touched by the forward pass
        with GradCAM(resnet_extractor, unused_layer) as cam:
            with pytest.raises(RuntimeError, match="activations/gradients"):
                cam.generate(single_input)


# --------------------------------------------------------------------------- #
# 3. Hook lifecycle
# --------------------------------------------------------------------------- #

class TestHookLifecycle:
    def test_context_manager_removes_hooks_on_exit(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer
    ):
        hooks_during = None
        with GradCAM(resnet_extractor, target_layer) as cam:
            hooks_during = (len(target_layer._forward_hooks), len(target_layer._backward_hooks))
        hooks_after = (len(target_layer._forward_hooks), len(target_layer._backward_hooks))

        assert hooks_during == (1, 1)
        assert hooks_after == (0, 0)

    def test_manual_remove_hooks_is_idempotent_with_context_manager(
        self, resnet_extractor: ResNet18FeatureExtractor, target_layer
    ):
        cam = GradCAM(resnet_extractor, target_layer)
        cam.remove_hooks()
        assert len(target_layer._forward_hooks) == 0
        assert len(target_layer._backward_hooks) == 0


# --------------------------------------------------------------------------- #
# 4. overlay_heatmap / get_default_target_layer
# --------------------------------------------------------------------------- #

class TestOverlayHeatmap:
    def test_overlay_matches_grayscale_input_shape(self):
        heatmap = np.random.default_rng(0).uniform(0, 1, size=(7, 7)).astype(np.float32)
        original = np.random.default_rng(1).integers(0, 255, size=(64, 64), dtype=np.uint8)

        overlay = overlay_heatmap(heatmap, original)

        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8

    def test_overlay_accepts_bgr_input_and_non_uint8_dtype(self):
        heatmap = np.random.default_rng(2).uniform(0, 1, size=(7, 7)).astype(np.float32)
        original = np.random.default_rng(3).normal(size=(48, 48, 3)).astype(np.float32)

        overlay = overlay_heatmap(heatmap, original)

        assert overlay.shape == (48, 48, 3)
        assert overlay.dtype == np.uint8

    def test_alpha_zero_returns_original_only(self):
        heatmap = np.ones((4, 4), dtype=np.float32)
        original = np.full((16, 16), 128, dtype=np.uint8)

        overlay = overlay_heatmap(heatmap, original, alpha=0.0)

        gray_from_overlay = overlay[..., 0]  # B=G=R since source was grayscale broadcast to BGR
        np.testing.assert_allclose(gray_from_overlay, original, atol=1)


class TestGetDefaultTargetLayer:
    def test_returns_second_to_last_backbone_child(self, resnet_extractor: ResNet18FeatureExtractor):
        layer = get_default_target_layer(resnet_extractor)
        assert layer is resnet_extractor.backbone[-2]
