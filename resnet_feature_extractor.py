"""
ResNet18-based feature extractor for MRI slices / volumes.

Uses a pretrained (ImageNet) ResNet18 backbone with the classification head
stripped off, tapping the `avgpool` output directly to produce a 512-D
feature vector per 2D input image. Early convolutional weights are frozen
so only downstream layers (if any are added later) would be trained.

Two usage modes:
    - Per-slice: input (B, 3, 224, 224) -> output (B, 512)
    - Per-volume: input (B, S, 3, 224, 224) -> per-slice features are
      extracted with the same backbone, then averaged over the slice
      dimension into a single (B, 512) volume embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18FeatureExtractor(nn.Module):
    """Frozen, pretrained ResNet18 backbone that outputs 512-D feature vectors.

    Args:
        freeze_backbone: If True (default), all backbone parameters have
            `requires_grad = False` set, so the pretrained weights are not
            updated during training (feature extraction / linear-probe use).
    """

    FEATURE_DIM = 512

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Strip the final fc layer; keep everything up to and including
        # avgpool. children() order for torchvision resnet18 is:
        # conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.freeze_backbone = freeze_backbone

    def train(self, mode: bool = True) -> "ResNet18FeatureExtractor":
        """Override train() so the frozen backbone always stays in eval mode
        (keeps BatchNorm running stats fixed) even if the parent module is
        switched to train() for other components."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward_slice(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of 2D images.

        Args:
            x: Tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor of shape (B, 512).
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(
                f"Expected input shape (B, 3, H, W), got {tuple(x.shape)}"
            )

        features = self.backbone(x)  # (B, 512, 1, 1)
        return torch.flatten(features, start_dim=1)  # (B, 512)

    def forward_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a single global embedding per multi-slice MRI volume by
        averaging per-slice features.

        Args:
            x: Tensor of shape (B, S, 3, 224, 224), where S is the number of
               slices in the volume.

        Returns:
            Tensor of shape (B, 512): mean-pooled feature vector per volume.
        """
        if x.dim() != 5 or x.shape[2] != 3:
            raise ValueError(
                f"Expected input shape (B, S, 3, H, W), got {tuple(x.shape)}"
            )

        batch_size, num_slices = x.shape[0], x.shape[1]

        # Fold slices into the batch dimension so the backbone runs once
        # over all slices, then unfold and average.
        flat_slices = x.reshape(batch_size * num_slices, *x.shape[2:])
        slice_features = self.forward_slice(flat_slices)  # (B*S, 512)

        slice_features = slice_features.reshape(batch_size, num_slices, self.FEATURE_DIM)
        return slice_features.mean(dim=1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatches to per-slice or per-volume extraction based on input rank.

        - 4D input (B, 3, 224, 224)    -> forward_slice
        - 5D input (B, S, 3, 224, 224) -> forward_volume
        """
        if x.dim() == 4:
            return self.forward_slice(x)
        elif x.dim() == 5:
            return self.forward_volume(x)
        else:
            raise ValueError(
                f"Expected 4D (B,3,H,W) or 5D (B,S,3,H,W) input, got {x.dim()}D"
            )


if __name__ == "__main__":
    torch.manual_seed(0)

    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    extractor.eval()

    trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    total = sum(p.numel() for p in extractor.parameters())
    print(f"Trainable params: {trainable} / {total} (expect 0 trainable when frozen)")

    # --- Per-slice test: (B, 3, 224, 224) -> (B, 512) ---
    dummy_slices = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        slice_features = extractor(dummy_slices)
    print(f"Per-slice input {tuple(dummy_slices.shape)} -> output {tuple(slice_features.shape)}")
    assert slice_features.shape == (4, 512)

    # --- Per-volume test: (B, S, 3, 224, 224) -> (B, 512) ---
    dummy_volume = torch.randn(2, 10, 3, 224, 224)  # 2 volumes, 10 slices each
    with torch.no_grad():
        volume_features = extractor(dummy_volume)
    print(f"Per-volume input {tuple(dummy_volume.shape)} -> output {tuple(volume_features.shape)}")
    assert volume_features.shape == (2, 512)

    # Sanity check: manually average slice features and compare to forward_volume
    with torch.no_grad():
        manual_features = extractor.forward_slice(dummy_volume[0])  # (10, 512)
        manual_mean = manual_features.mean(dim=0)  # (512,)
    torch.testing.assert_close(manual_mean, volume_features[0], rtol=1e-4, atol=1e-4)
    print("Manual per-slice averaging matches forward_volume output. All tests passed.")
