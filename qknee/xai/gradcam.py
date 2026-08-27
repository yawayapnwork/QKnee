"""
Grad-CAM explainability for the Q-Knee ResNet18 backbone.

Targets the final convolutional block (`layer4`, immediately before
`avgpool`) of `resnet_feature_extractor.ResNet18FeatureExtractor` and
produces a spatial activation heatmap showing which anatomical regions of
an MRI slice most influenced the extracted 512-D feature vector.

Because the backbone is a frozen feature extractor, generating a heatmap
that actually explains a *prediction* (rather than just the embedding)
requires backpropagating from that prediction, not from the embedding
itself. Two targeting modes are supported:

    - Custom `target_fn` (preferred, and what `PipelineRunner.explain()`
      uses): pass any callable `(B, 512) -> scalar tensor` that continues
      the forward computation through to a real model output — e.g.
      `qknee.models.pipeline.PipelineRunner`'s risk-score target, which
      chains the 512-D embedding through the differentiable PCA projection
      and the VQC to the predicted tear-risk probability, so the resulting
      heatmap highlights the regions that actually drove *that* prediction.
    - Default ("embedding energy"): backprops the squared L2 norm of the
      512-D embedding when no `target_fn` is given, highlighting regions
      that drive the overall feature representation most strongly. This is
      a reasonable fallback for exploring the backbone in isolation, but is
      not class/prediction-discriminative — prefer a real `target_fn`
      whenever a trained downstream head is available.

Uses OpenCV for colormap generation and overlay compositing (no TorchCAM
dependency required).
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger
from qknee.models.resnet_extractor import ResNet18FeatureExtractor

logger = get_logger(__name__)
_config = load_config()

TargetFn = Callable[[torch.Tensor], torch.Tensor]


class GradCAM:
    """Grad-CAM hook manager for a single target convolutional layer.

    Usage:
        extractor = ResNet18FeatureExtractor()
        with GradCAM(extractor, target_layer=extractor.backbone[7]) as cam:
            heatmap = cam.generate(input_tensor)  # (H, W) in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def generate(self, input_tensor: torch.Tensor, target_fn: Optional[TargetFn] = None) -> np.ndarray:
        """Runs a forward + backward pass and produces the Grad-CAM heatmap.

        Args:
            input_tensor: (1, 3, 224, 224) tensor (single image; batch size 1
                keeps the returned heatmap unambiguous).
            target_fn: Optional callable mapping the model's output
                (whatever `self.model(input_tensor)` returns) to a scalar
                tensor to backpropagate — e.g. a predicted risk probability
                continued through a downstream classifier. Must return a
                single-element tensor (0-dim, or `.numel() == 1`); anything
                else raises a `ValueError` naming the offending shape rather
                than failing inside `.backward()`. Defaults to the squared
                L2 norm of the output (embedding-energy Grad-CAM) when omitted.

        Returns:
            (H, W) numpy array, normalized to [0, 1], at the target layer's
            native spatial resolution (typically 7x7 for ResNet18 layer4 on
            a 224x224 input) — call `overlay_heatmap` to resize + blend it
            onto the original image.
        """
        if input_tensor.shape[0] != 1:
            raise ValueError(f"GradCAM.generate expects batch size 1, got {input_tensor.shape[0]}")

        self.model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.clone().requires_grad_(True)

        was_training = self.model.training
        self.model.eval()
        try:
            output = self.model(input_tensor)
            target = target_fn(output) if target_fn is not None else output.pow(2).sum()
            if target.numel() != 1:
                raise ValueError(
                    f"target_fn must return a single-element (scalar) tensor to backpropagate "
                    f"from, got shape {tuple(target.shape)}. Reduce it (e.g. `.squeeze()` or "
                    f"`.mean()`) before returning it from target_fn."
                )
            target.squeeze().backward()
        finally:
            self.model.train(was_training)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "No activations/gradients captured — is target_layer actually "
                "part of the forward path of `model`?"
            )

        # Global-average-pool the gradients per channel -> per-channel importance weight.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1)).squeeze(0)  # (H, W)

        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min).item() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    alpha: float = _config.gradcam.alpha,
    colormap: int = getattr(cv2, _config.gradcam.colormap),
) -> np.ndarray:
    """Resizes `heatmap` to match `original_image` and alpha-blends a
    color-mapped version on top.

    Args:
        heatmap: (h, w) array in [0, 1], typically the raw Grad-CAM output.
        original_image: (H, W) grayscale or (H, W, 3) BGR uint8 MRI slice.
        alpha: Blend weight of the heatmap (0 = original only, 1 = heatmap only).
        colormap: OpenCV colormap constant.

    Returns:
        (H, W, 3) uint8 BGR image with the heatmap overlaid.
    """
    original_image = np.asarray(original_image)
    if original_image.dtype != np.uint8:
        norm = original_image.astype(np.float32)
        min_val, max_val = float(norm.min()), float(norm.max())
        norm = (norm - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(norm)
        original_image = (norm * 255).astype(np.uint8)

    if original_image.ndim == 2:
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        original_bgr = original_image

    height, width = original_bgr.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = (np.clip(resized_heatmap, 0, 1) * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

    return cv2.addWeighted(color_heatmap, alpha, original_bgr, 1 - alpha, 0)


def get_default_target_layer(extractor: ResNet18FeatureExtractor) -> nn.Module:
    """Returns `layer4` (the final convolutional block, right before
    `avgpool`) of a `ResNet18FeatureExtractor`'s backbone."""
    # backbone children order: conv1, bn1, relu, maxpool, layer1, layer2,
    # layer3, layer4, avgpool -> layer4 is index -2.
    return extractor.backbone[-2]


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()
    torch.manual_seed(0)

    output_dir = _config.paths.eval_output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    extractor.eval()
    target_layer = get_default_target_layer(extractor)
    logger.info(
        "Target layer for Grad-CAM: %s (ResNet18 layer4, %d params)",
        target_layer.__class__.__name__,
        sum(p.numel() for p in target_layer.parameters()),
    )

    # --- Dummy MRI-like slice (grayscale ring pattern so the CAM has visible structure) ---
    yy, xx = np.mgrid[0:224, 0:224]
    center = 112
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    dummy_slice = (255 * np.clip(1 - np.abs(radius - 70) / 40, 0, 1)).astype(np.uint8)

    rgb_slice = cv2.cvtColor(dummy_slice, cv2.COLOR_GRAY2RGB)
    input_tensor = torch.from_numpy(rgb_slice).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    with GradCAM(extractor, target_layer) as cam:
        heatmap = cam.generate(input_tensor)

    logger.info(
        "Raw Grad-CAM shape: %s (native layer4 resolution), range [%.3f, %.3f]",
        heatmap.shape, heatmap.min(), heatmap.max(),
    )
    assert heatmap.ndim == 2
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0

    overlaid = overlay_heatmap(heatmap, dummy_slice)
    assert overlaid.shape == (224, 224, 3)

    out_path = output_dir / "gradcam_demo_overlay.png"
    cv2.imwrite(str(out_path), overlaid)
    logger.info("Saved Grad-CAM overlay to %s", out_path.resolve())
    logger.info("Grad-CAM smoke test passed.")
