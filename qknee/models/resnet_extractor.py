"""
ResNet18-based feature extractor for MRI slices / volumes.

Uses a pretrained (ImageNet) ResNet18 backbone with the classification head
stripped off, tapping the `avgpool` output directly to produce a 512-D
feature vector per 2D input image. Early convolutional weights are frozen
so only downstream layers (if any are added later) would be trained.

Two usage modes:
    - Per-slice: input (B, 3, 224, 224) -> output (B, 512)
    - Per-volume: input (B, S, 3, 224, 224) -> per-slice features are
      extracted with the same backbone, then aggregated over the (variable-
      length) slice dimension into a single (B, 512) volume embedding via
      `config.resnet.volumetric_aggregation` ("mean" | "attention" |
      "topk_max" — see `AttentionPooling` / `topk_max_aggregate` below).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from qknee.config.loader import load_config
from qknee.config.logging_config import get_logger

_config = load_config()
logger = get_logger(__name__)

VOLUMETRIC_AGGREGATIONS = ("mean", "attention", "topk_max")


class AttentionPooling(nn.Module):
    """Gated-attention pooling (Ilse et al., 2018 style) over a variable
    number of per-slice feature vectors: a small MLP scores each slice's
    512-D feature vector, scores are softmax-normalized across the slice
    dimension, and the volume embedding is the resulting convex combination
    of slice features. Unlike plain mean-pooling, slices contributing more
    diagnostically relevant signal can receive a larger weight.

    Args:
        feature_dim: Dimensionality of each per-slice feature vector.
        hidden_dim: Width of the scoring MLP's hidden layer.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slice_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            slice_features: `(B, S, feature_dim)`.

        Returns:
            `(pooled, weights)` where `pooled` is `(B, feature_dim)` and
            `weights` is `(B, S)` (softmax-normalized attention weights,
            summing to 1 across the slice dimension — useful for
            inspecting which slices drove the volume embedding).
        """
        scores = self.score(slice_features)  # (B, S, 1)
        weights = torch.softmax(scores, dim=1)  # (B, S, 1)
        pooled = (slice_features * weights).sum(dim=1)  # (B, feature_dim)
        return pooled, weights.squeeze(-1)


def topk_max_aggregate(slice_features: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k maximum feature aggregation across the slice dimension: for
    each of the `feature_dim` channels independently, takes the `k`
    largest per-slice values and averages them. This is a standard
    multiple-instance-learning pooling rule — it lets the volume embedding
    be driven by whichever slices most strongly activate each feature
    channel, rather than diluting a strong signal across every slice the
    way plain mean-pooling does.

    Args:
        slice_features: `(B, S, feature_dim)`.
        k: Number of top slices to average per channel. Clamped to `S` if
            `k > S` (a short volume still aggregates correctly rather than
            raising).

    Returns:
        `(B, feature_dim)`.
    """
    num_slices = slice_features.shape[1]
    k = min(k, num_slices)
    top_values, _ = torch.topk(slice_features, k, dim=1)  # (B, k, feature_dim)
    return top_values.mean(dim=1)  # (B, feature_dim)


class ResNet18FeatureExtractor(nn.Module):
    """Frozen, pretrained ResNet18 backbone that outputs 512-D feature vectors.

    Args:
        freeze_backbone: If True (default, from `config.yaml`'s
            `resnet.freeze_backbone`), all backbone parameters have
            `requires_grad = False` set, so the pretrained weights are not
            updated during training (feature extraction / linear-probe use).
        volumetric_aggregation: How `forward_volume` collapses a volume's
            variable-length slice dimension into one 512-D embedding —
            `"mean"` (plain average), `"attention"` (learned
            `AttentionPooling` over slices), or `"topk_max"` (per-channel
            top-`topk_k` max, see `topk_max_aggregate`). Defaults to
            `config.yaml`'s `resnet.volumetric_aggregation`.
        topk_k: Slices averaged per channel when
            `volumetric_aggregation == "topk_max"`. Defaults to
            `config.yaml`'s `resnet.topk_k`.
    """

    FEATURE_DIM = _config.resnet.feature_dim

    def __init__(
        self,
        freeze_backbone: bool = _config.resnet.freeze_backbone,
        volumetric_aggregation: str = _config.resnet.volumetric_aggregation,
        topk_k: int = _config.resnet.topk_k,
    ):
        super().__init__()

        if volumetric_aggregation not in VOLUMETRIC_AGGREGATIONS:
            raise ValueError(
                f"volumetric_aggregation must be one of {VOLUMETRIC_AGGREGATIONS}, "
                f"got {volumetric_aggregation!r}"
            )
        if topk_k < 1:
            raise ValueError(f"topk_k must be >= 1, got {topk_k}")

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Strip the final fc layer; keep everything up to and including
        # avgpool. children() order for torchvision resnet18 is:
        # conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # `channels_last` (NHWC) memory format lets modern CPU BLAS/MKL-DNN
        # kernels (and cuDNN on GPU) use their vectorized/AVX-512 conv
        # paths more effectively than the default `channels_first` (NCHW)
        # layout — a free win for a frozen, inference-only conv backbone
        # since there's no training-time weight-update path that would
        # need to round-trip the conversion. Purely a memory-layout hint:
        # never changes numerics, so this is safe regardless of whether
        # Grad-CAM's hook-based gradient path (which shares this same
        # `self.backbone`) is in play.
        self.backbone = self.backbone.to(memory_format=torch.channels_last)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.freeze_backbone = freeze_backbone
        self.volumetric_aggregation = volumetric_aggregation
        self.topk_k = topk_k
        self.attention_pool = (
            AttentionPooling(self.FEATURE_DIM) if volumetric_aggregation == "attention" else None
        )
        self._last_attention_weights: Optional[torch.Tensor] = None

        # Lazily-built TorchScript-traced + `optimize_for_inference`'d copy
        # of `self.backbone`, used only by `forward_slice` when gradients
        # are disabled (see there) — built once, on first qualifying call,
        # from a real input's shape so the trace matches production usage
        # exactly. `None` until built, and permanently `None` again (with
        # a logged warning) if tracing/optimization ever fails — this path
        # is pure speedup, never required for correctness, so any failure
        # just means every future call keeps using the eager backbone
        # rather than raising.
        self._traced_backbone: Optional[torch.jit.ScriptModule] = None
        self._traced_backbone_failed = False

    def train(self, mode: bool = True) -> "ResNet18FeatureExtractor":
        """Override train() so the frozen backbone always stays in eval mode
        (keeps BatchNorm running stats fixed) even if the parent module is
        switched to train() for other components."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _get_traced_backbone(self, example: torch.Tensor) -> Optional[torch.jit.ScriptModule]:
        """Builds (once) and returns the TorchScript-traced +
        `optimize_for_inference`'d backbone, or `None` if that isn't
        possible/safe right now. Only ever called from `forward_slice`
        when gradients are already disabled (see there) — a traced,
        optimized graph doesn't support the autograd tracking Grad-CAM's
        hook-based path needs, so this must never be reached from a
        gradient-requiring forward pass."""
        if self._traced_backbone_failed:
            return None
        if self._traced_backbone is not None:
            return self._traced_backbone

        try:
            traced = torch.jit.trace(self.backbone, example.to(memory_format=torch.channels_last))
            traced = torch.jit.optimize_for_inference(traced)
            self._traced_backbone = traced
            logger.info("Traced + optimized ResNet18 backbone for inference (TorchScript).")
        except Exception as exc:  # noqa: BLE001 - tracing is a pure speedup; never fatal
            logger.warning("TorchScript trace/optimize_for_inference failed (%s); using eager backbone.", exc)
            self._traced_backbone_failed = True
            return None

        return self._traced_backbone

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

        x = x.to(memory_format=torch.channels_last)

        # The traced/optimized backbone strips the graph structure Grad-CAM's
        # forward/backward hooks (registered on `self.backbone`'s named
        # submodules — see qknee.xai.gradcam.GradCAM) rely on, and can't be
        # backpropagated through — so it's only used when this call is
        # already gradient-free (every pure-inference caller: `/predict`,
        # `/explain`'s prediction step, the dashboard/workstation). Grad-CAM's
        # own forward pass always runs with gradients enabled, so it
        # transparently keeps using the eager `self.backbone` below.
        if not torch.is_grad_enabled():
            traced_backbone = self._get_traced_backbone(x)
            if traced_backbone is not None:
                features = traced_backbone(x)  # (B, 512, 1, 1)
                return torch.flatten(features, start_dim=1)  # (B, 512)

        features = self.backbone(x)  # (B, 512, 1, 1)
        return torch.flatten(features, start_dim=1)  # (B, 512)

    def forward_volume(self, x: torch.Tensor) -> torch.Tensor:
        """Extract a single global embedding per multi-slice MRI volume by
        aggregating per-slice features across the (variable-length) slice
        dimension, via `self.volumetric_aggregation`.

        Args:
            x: Tensor of shape (B, S, 3, 224, 224), where S is the number of
               slices in the volume — S may vary freely between calls (e.g.
               20 to 45 slices per study), since the backbone runs
               per-slice and aggregation reduces over whatever S is passed.

        Returns:
            Tensor of shape (B, 512): aggregated feature vector per volume.
        """
        if x.dim() != 5 or x.shape[2] != 3:
            raise ValueError(
                f"Expected input shape (B, S, 3, H, W), got {tuple(x.shape)}"
            )

        batch_size, num_slices = x.shape[0], x.shape[1]

        # Fold slices into the batch dimension so the backbone runs once
        # over all slices, then unfold and aggregate.
        flat_slices = x.reshape(batch_size * num_slices, *x.shape[2:])
        slice_features = self.forward_slice(flat_slices)  # (B*S, 512)

        slice_features = slice_features.reshape(batch_size, num_slices, self.FEATURE_DIM)

        if self.volumetric_aggregation == "mean":
            self._last_attention_weights = None
            return slice_features.mean(dim=1)  # (B, 512)
        elif self.volumetric_aggregation == "topk_max":
            self._last_attention_weights = None
            return topk_max_aggregate(slice_features, self.topk_k)  # (B, 512)
        else:  # "attention"
            pooled, weights = self.attention_pool(slice_features)  # (B, 512), (B, S)
            self._last_attention_weights = weights
            return pooled

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
    """Exploratory / outside the judged PRD scope — inert behind `config.yaml`'s `resnet.backend_engine` defaulting to `"pytorch"`, not `"onnx"`.

    onnxruntime-accelerated drop-in for `ResNet18FeatureExtractor`'s
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

    extractor = ResNet18FeatureExtractor(freeze_backbone=True, volumetric_aggregation="mean")
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
    logger.info("Manual per-slice averaging matches forward_volume (mean) output.")

    # --- Attention pooling + top-k max aggregation, over variable slice counts ---
    for aggregation in ("attention", "topk_max"):
        aggregator = ResNet18FeatureExtractor(
            freeze_backbone=True, volumetric_aggregation=aggregation, topk_k=3,
        )
        aggregator.eval()
        for num_slices in (5, 20, 45):  # variable volume depth, per the RSNA spec
            volume = torch.randn(2, num_slices, 3, 224, 224)
            with torch.no_grad():
                pooled = aggregator(volume)
            assert pooled.shape == (2, 512), f"{aggregation}, S={num_slices}: got {tuple(pooled.shape)}"
        logger.info("volumetric_aggregation=%r handles variable slice counts -> (B, 512).", aggregation)

    logger.info("All resnet_extractor.py smoke tests passed.")
