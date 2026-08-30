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

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

_config = load_config()
logger = get_logger(__name__)


class ResNet18FeatureExtractor(nn.Module):
    """Frozen, pretrained ResNet18 backbone that outputs 512-D feature vectors.

    Args:
        freeze_backbone: If True (default, from `config.yaml`'s
            `resnet.freeze_backbone`), all backbone parameters have
            `requires_grad = False` set, so the pretrained weights are not
            updated during training (feature extraction / linear-probe use).
    """

    FEATURE_DIM = _config.resnet.feature_dim

    def __init__(self, freeze_backbone: bool = _config.resnet.freeze_backbone):
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

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Backend-agnostic alias for `forward`/`__call__` — the interface
        `PipelineRunner` (and `ONNXFeatureExtractor`) code against so either
        backend is a plug-and-play swap regardless of which is selected via
        `config.resnet.backend_engine`. Accepts either a 4D per-slice batch
        or a 5D per-volume batch (see `forward`)."""
        return self(batch)


class ONNXFeatureExtractor:
    """onnxruntime-accelerated drop-in for `ResNet18FeatureExtractor`'s
    inference path (`forward_slice` / `forward_volume` / `forward` /
    `__call__`), for deployments that prefer ONNX Runtime's graph-optimized
    CPU/GPU execution over eager PyTorch.

    Load a graph exported by `scripts/export_onnx.py` — a
    `(B, 3, 224, 224) -> (B, 512)` ONNX model with a dynamic batch axis.

    Not an `nn.Module` — no PyTorch autograd runs through it (ONNX Runtime
    is inference-only here), so it's for the frozen-backbone inference
    path only; Grad-CAM's `explain()` needs real PyTorch gradients through
    the backbone and must keep using `ResNet18FeatureExtractor`.

    Args:
        onnx_path: Path to a `.onnx` file exported by
            `scripts/export_onnx.py`.
        providers: onnxruntime execution providers, in priority order.
            Defaults to GPU-if-available (`CUDAExecutionProvider`),
            falling back to `CPUExecutionProvider` — pass an explicit list
            to pin one provider regardless of what's installed/available.
        intra_op_num_threads: Threads used to parallelize *within* a single
            op (e.g. one conv's matmul) on `CPUExecutionProvider`. Defaults
            to `os.cpu_count()` — ONNX Runtime's own default is a good
            general choice, but this backbone is a fixed, always-CPU-bound
            frozen-inference workload, so pinning it explicitly to all
            available cores avoids under-using the host in a container with
            a restrictive default thread count.
        inter_op_num_threads: Threads used to run independent op subgraphs
            *in parallel* with each other. Left at 1 (ONNX Runtime's
            default) since `execution_mode` is sequential below; only
            relevant if a caller overrides `execution_mode` to parallel.
        execution_mode: `"sequential"` (default) runs the graph's ops one
            at a time, each fanning out across `intra_op_num_threads` —
            the better choice for this backbone, a single linear chain of
            conv/BN/relu blocks with no independent branches to run
            concurrently. `"parallel"` is exposed for callers with a
            different graph shape.
    """

    FEATURE_DIM = _config.resnet.feature_dim

    def __init__(
        self,
        onnx_path: Union[str, Path],
        providers: Optional[List[str]] = None,
        intra_op_num_threads: Optional[int] = None,
        inter_op_num_threads: int = 1,
        execution_mode: str = "sequential",
    ) -> None:
        import os

        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. Export one first via "
                "`python scripts/export_onnx.py`."
            )

        if providers is None:
            available = ort.get_available_providers()
            providers = (["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else [])
            providers.append("CPUExecutionProvider")

        # Optimized CPU thread pooling: pin intra-op parallelism to all
        # available cores (rather than leaving it at ONNX Runtime's
        # environment-dependent default) and enable the full graph
        # optimization level (op fusion, constant folding, layout
        # optimization), since this is a fixed inference-only graph with
        # no further training/export step downstream that could need the
        # unoptimized graph preserved.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads or (os.cpu_count() or 1)
        session_options.inter_op_num_threads = inter_op_num_threads
        session_options.execution_mode = (
            ort.ExecutionMode.ORT_PARALLEL if execution_mode == "parallel" else ort.ExecutionMode.ORT_SEQUENTIAL
        )
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.onnx_path = onnx_path
        self.providers = providers
        self.session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name

        logger.info(
            "ONNXFeatureExtractor loaded %s (active provider: %s, intra_op_num_threads=%d)",
            onnx_path, self.session.get_providers()[0], session_options.intra_op_num_threads,
        )

    def forward_slice(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from a batch of 2D images via ONNX Runtime.

        Args:
            x: Tensor of shape (B, 3, 224, 224), on any device — moved to
                CPU/numpy for the ONNX Runtime session regardless of
                `x`'s original device, then returned on that same device.

        Returns:
            Tensor of shape (B, 512).
        """
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape (B, 3, H, W), got {tuple(x.shape)}")

        input_array = x.detach().cpu().numpy().astype(np.float32)
        output_array = self.session.run([self._output_name], {self._input_name: input_array})[0]
        return torch.from_numpy(output_array).to(x.device)

    def forward_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a single global embedding per multi-slice MRI volume by
        averaging per-slice ONNX Runtime features — same slice-averaging
        semantics as `ResNet18FeatureExtractor.forward_volume`.

        Args:
            x: Tensor of shape (B, S, 3, 224, 224).

        Returns:
            Tensor of shape (B, 512).
        """
        if x.dim() != 5 or x.shape[2] != 3:
            raise ValueError(f"Expected input shape (B, S, 3, H, W), got {tuple(x.shape)}")

        batch_size, num_slices = x.shape[0], x.shape[1]
        flat_slices = x.reshape(batch_size * num_slices, *x.shape[2:])
        slice_features = self.forward_slice(flat_slices)  # (B*S, 512)

        slice_features = slice_features.reshape(batch_size, num_slices, self.FEATURE_DIM)
        return slice_features.mean(dim=1)  # (B, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatches to per-slice or per-volume extraction based on input
        rank — mirrors `ResNet18FeatureExtractor.forward`."""
        if x.dim() == 4:
            return self.forward_slice(x)
        elif x.dim() == 5:
            return self.forward_volume(x)
        else:
            raise ValueError(f"Expected 4D (B,3,H,W) or 5D (B,S,3,H,W) input, got {x.dim()}D")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        """Backend-agnostic alias for `forward`/`__call__` — the interface
        `PipelineRunner` codes against so `ResNet18FeatureExtractor` and
        `ONNXFeatureExtractor` are a plug-and-play swap regardless of which
        is selected via `config.resnet.backend_engine`. Accepts either a 4D
        per-slice batch or a 5D per-volume batch (see `forward`)."""
        return self(batch)

    def eval(self) -> "ONNXFeatureExtractor":
        """No-op, provided for interface parity with
        `ResNet18FeatureExtractor.eval()` — an ONNX Runtime inference
        session has no train/eval mode to toggle."""
        return self

    def to(self, device: Union[str, torch.device]) -> "ONNXFeatureExtractor":
        """No-op, provided for interface parity with `nn.Module.to()`.
        ONNX Runtime's device placement is controlled by `providers` at
        construction time (e.g. `CUDAExecutionProvider`), not by moving
        tensors onto a device here."""
        return self


if __name__ == "__main__":
    from qknee.config.logging_config import setup_logging

    setup_logging()

    torch.manual_seed(0)

    extractor = ResNet18FeatureExtractor(freeze_backbone=True)
    extractor.eval()

    trainable = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    total = sum(p.numel() for p in extractor.parameters())
    logger.info("Trainable params: %d / %d (expect 0 trainable when frozen)", trainable, total)

    # --- Per-slice test: (B, 3, 224, 224) -> (B, 512) ---
    dummy_slices = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        slice_features = extractor(dummy_slices)
    logger.info("Per-slice input %s -> output %s", tuple(dummy_slices.shape), tuple(slice_features.shape))
    assert slice_features.shape == (4, 512)

    # --- Per-volume test: (B, S, 3, 224, 224) -> (B, 512) ---
    dummy_volume = torch.randn(2, 10, 3, 224, 224)  # 2 volumes, 10 slices each
    with torch.no_grad():
        volume_features = extractor(dummy_volume)
    logger.info("Per-volume input %s -> output %s", tuple(dummy_volume.shape), tuple(volume_features.shape))
    assert volume_features.shape == (2, 512)

    # Sanity check: manually average slice features and compare to forward_volume
    with torch.no_grad():
        manual_features = extractor.forward_slice(dummy_volume[0])  # (10, 512)
        manual_mean = manual_features.mean(dim=0)  # (512,)
    torch.testing.assert_close(manual_mean, volume_features[0], rtol=1e-4, atol=1e-4)
    logger.info("Manual per-slice averaging matches forward_volume output. All tests passed.")
