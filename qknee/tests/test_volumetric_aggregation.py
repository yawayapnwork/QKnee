"""
Unit tests for 3D volumetric slice-sequence aggregation
(`qknee.models.resnet_extractor`) and multi-plane series fusion
(`qknee.data.dataset.MultiPlaneEmbeddingFusion`):

    1. Arbitrary (Batch, Slices, Channels, H, W) volumes collapse correctly
       into (Batch, 4) quantum-ready state vectors, for every
       `volumetric_aggregation` mode ("mean", "attention", "topk_max") and
       across variable slice depths (e.g. 20-45 slices/study).
    2. Repeated forward passes over such volumes do not leak memory.
    3. Multi-plane 4-D embeddings (Sagittal/Coronal/Axial) fuse correctly
       into a single 4-D embedding via both fusion methods, including when
       a study is missing one or more planes.
"""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np
import pytest
import torch

from qknee.data.dataset import RSNA_PLANES, MultiPlaneEmbeddingFusion
from qknee.models.pca_reducer import QuantumDimReducer
from qknee.models.qknee_model import PCAProjectionLayer
from qknee.models.resnet_extractor import AttentionPooling, ResNet18FeatureExtractor, topk_max_aggregate

VOLUMETRIC_AGGREGATIONS = ("mean", "attention", "topk_max")


@pytest.fixture(scope="module")
def pca_layer() -> PCAProjectionLayer:
    """A real, fitted 512 -> 4 bottleneck (frozen buffers, no gradient
    dependency on the extractor), shared across this module's tests."""
    rng = np.random.default_rng(0)
    reducer = QuantumDimReducer().fit(rng.normal(size=(200, 512)).astype(np.float32))
    return PCAProjectionLayer.from_reducer(reducer)


class TestVolumeToQuantumVectorCollapse:
    """(Batch, Slices, Channels, H, W) -> ResNet18 volumetric aggregation ->
    (Batch, 512) -> PCAProjectionLayer -> (Batch, 4)."""

    @pytest.mark.parametrize("aggregation", VOLUMETRIC_AGGREGATIONS)
    @pytest.mark.parametrize("batch_size,num_slices", [(1, 20), (2, 32), (3, 45), (2, 7)])
    def test_variable_shape_collapses_to_batch_by_4(
        self, aggregation, batch_size, num_slices, pca_layer,
    ):
        torch.manual_seed(0)
        extractor = ResNet18FeatureExtractor(
            freeze_backbone=True, volumetric_aggregation=aggregation, topk_k=5,
        )
        extractor.eval()

        volume = torch.rand(batch_size, num_slices, 3, 224, 224)
        with torch.no_grad():
            volume_features = extractor(volume)  # (B, 512)
            quantum_vector = pca_layer(volume_features)  # (B, 4)

        assert volume_features.shape == (batch_size, 512)
        assert quantum_vector.shape == (batch_size, 4)
        assert torch.isfinite(quantum_vector).all()

    def test_topk_k_larger_than_num_slices_does_not_raise(self, pca_layer):
        """A short volume (fewer slices than topk_k) must still aggregate
        correctly — topk_max_aggregate clamps k to the available slices."""
        torch.manual_seed(0)
        extractor = ResNet18FeatureExtractor(
            freeze_backbone=True, volumetric_aggregation="topk_max", topk_k=50,
        )
        extractor.eval()

        volume = torch.rand(2, 3, 3, 224, 224)  # only 3 slices, topk_k=50
        with torch.no_grad():
            volume_features = extractor(volume)
            quantum_vector = pca_layer(volume_features)

        assert quantum_vector.shape == (2, 4)

    def test_attention_pooling_weights_sum_to_one(self):
        torch.manual_seed(0)
        pooling = AttentionPooling(feature_dim=512)
        slice_features = torch.randn(3, 12, 512)
        pooled, weights = pooling(slice_features)

        assert pooled.shape == (3, 512)
        assert weights.shape == (3, 12)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3), rtol=1e-5, atol=1e-5)

    def test_attention_differs_from_mean_pooling(self):
        """A trained (non-uniform) attention MLP should generally weight
        slices unequally, producing a different pooled vector than a plain
        mean — sanity-checks that attention pooling isn't secretly a no-op
        mean under the hood."""
        torch.manual_seed(1)
        slice_features = torch.randn(1, 16, 512)

        pooling = AttentionPooling(feature_dim=512)
        # Push the scoring MLP's weights away from zero-init so its scores
        # aren't (numerically) uniform, which would make attention and mean
        # coincide by construction rather than by the input distribution.
        with torch.no_grad():
            for param in pooling.parameters():
                param.add_(torch.randn_like(param) * 0.5)
        attention_pooled, _ = pooling(slice_features)
        mean_pooled = slice_features.mean(dim=1)

        assert not torch.allclose(attention_pooled, mean_pooled, atol=1e-4)

    def test_topk_max_matches_manual_per_channel_topk(self):
        torch.manual_seed(2)
        slice_features = torch.randn(2, 10, 512)
        k = 4

        aggregated = topk_max_aggregate(slice_features, k)
        manual = slice_features.topk(k, dim=1).values.mean(dim=1)

        torch.testing.assert_close(aggregated, manual)

    def test_gradients_flow_through_aggregation(self, pca_layer):
        """The 512-D volume embedding (and the 4-D quantum vector derived
        from it) must remain differentiable w.r.t. the input pixels, so
        Grad-CAM-style attribution works over volumetric inputs too."""
        torch.manual_seed(0)
        extractor = ResNet18FeatureExtractor(
            freeze_backbone=True, volumetric_aggregation="attention", topk_k=5,
        )
        extractor.train()  # backbone stays eval() internally (frozen), but grads still flow

        volume = torch.rand(2, 6, 3, 224, 224, requires_grad=True)
        volume_features = extractor(volume)
        quantum_vector = pca_layer(volume_features)
        quantum_vector.sum().backward()

        assert volume.grad is not None
        assert torch.isfinite(volume.grad).all()
        assert volume.grad.abs().sum() > 0


class TestVolumetricAggregationMemoryStability:
    """Repeated forward passes over volumetric inputs must not accumulate
    unbounded memory (e.g. from an accidentally retained autograd graph or
    a growing Python-side cache)."""

    @pytest.mark.parametrize("aggregation", VOLUMETRIC_AGGREGATIONS)
    def test_repeated_inference_does_not_grow_memory(self, aggregation):
        """Runs many volumetric forward passes and asserts net allocated
        memory (tracked via `tracemalloc`, scoped to this test only — not
        polluted by other tests' fixtures/tensors in the same session)
        stays well under one iteration's worth of input memory. A real leak
        (e.g. a retained autograd graph, or an aggregator accumulating
        state across calls) would grow roughly linearly with the number of
        iterations instead."""
        torch.manual_seed(0)
        extractor = ResNet18FeatureExtractor(
            freeze_backbone=True, volumetric_aggregation=aggregation, topk_k=5,
        )
        extractor.eval()

        def run_batch() -> None:
            volume = torch.rand(2, 24, 3, 224, 224)
            with torch.no_grad():
                _ = extractor(volume)
            del volume

        # Warm up: absorb any one-time allocations (e.g. lazily-initialized
        # internal buffers) before measuring.
        for _ in range(3):
            run_batch()
        gc.collect()

        tracemalloc.start()
        try:
            snapshot_before = tracemalloc.take_snapshot()
            n_iterations = 10
            for _ in range(n_iterations):
                run_batch()
            gc.collect()
            snapshot_after = tracemalloc.take_snapshot()
        finally:
            tracemalloc.stop()

        stats = snapshot_after.compare_to(snapshot_before, "lineno")
        net_growth_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)

        one_iteration_bytes = 2 * 24 * 3 * 224 * 224 * 4  # one volume's worth of float32 input
        assert net_growth_bytes < one_iteration_bytes, (
            f"{aggregation}: net memory growth over {n_iterations} iterations "
            f"({net_growth_bytes} bytes) suggests a leak — expected < "
            f"{one_iteration_bytes} bytes (one iteration's input size)."
        )

    def test_no_autograd_graph_retained_across_no_grad_calls(self):
        """Under `torch.no_grad()`, no autograd graph nodes should
        accumulate across repeated calls — a proxy for the classic
        "forgot to .detach()" memory leak in volumetric loops."""
        torch.manual_seed(0)
        extractor = ResNet18FeatureExtractor(
            freeze_backbone=True, volumetric_aggregation="attention", topk_k=5,
        )
        extractor.eval()

        for _ in range(5):
            volume = torch.rand(2, 15, 3, 224, 224)
            with torch.no_grad():
                output = extractor(volume)
            assert output.grad_fn is None
            assert not output.requires_grad
            del volume, output

        gc.collect()


class TestMultiPlaneEmbeddingFusion:
    @pytest.mark.parametrize("method", ["weighted_average", "linear_bottleneck"])
    def test_all_planes_present_fuses_to_batch_by_4(self, method):
        torch.manual_seed(0)
        fusion = MultiPlaneEmbeddingFusion(method=method)
        embeddings = {plane: torch.randn(5, 4) for plane in RSNA_PLANES}

        fused = fusion(embeddings)

        assert fused.shape == (5, 4)
        assert torch.isfinite(fused).all()

    @pytest.mark.parametrize("method", ["weighted_average", "linear_bottleneck"])
    def test_missing_plane_still_fuses(self, method):
        """Not every study has all three planes — fusion must handle a
        partial subset without raising."""
        torch.manual_seed(0)
        fusion = MultiPlaneEmbeddingFusion(method=method)
        embeddings = {"Sagittal": torch.randn(3, 4), "Axial": torch.randn(3, 4)}

        fused = fusion(embeddings)

        assert fused.shape == (3, 4)

    def test_weighted_average_is_convex_combination(self):
        torch.manual_seed(0)
        fusion = MultiPlaneEmbeddingFusion(method="weighted_average")
        embeddings = {plane: torch.randn(4, 4) for plane in RSNA_PLANES}

        fused = fusion(embeddings)
        stacked = torch.stack(list(embeddings.values()), dim=1)  # (B, 3, 4)

        assert (fused.unsqueeze(1) >= stacked.min(dim=1, keepdim=True).values - 1e-4).all()
        assert (fused.unsqueeze(1) <= stacked.max(dim=1, keepdim=True).values + 1e-4).all()

    def test_empty_embeddings_raises(self):
        fusion = MultiPlaneEmbeddingFusion(method="weighted_average")
        with pytest.raises(ValueError, match="empty"):
            fusion({})

    def test_unknown_plane_raises(self):
        fusion = MultiPlaneEmbeddingFusion(method="weighted_average")
        with pytest.raises(ValueError, match="Unknown plane"):
            fusion({"Oblique": torch.randn(2, 4)})

    def test_gradients_flow_through_both_methods(self):
        for method in ("weighted_average", "linear_bottleneck"):
            torch.manual_seed(0)
            fusion = MultiPlaneEmbeddingFusion(method=method)
            embeddings = {plane: torch.randn(2, 4, requires_grad=True) for plane in RSNA_PLANES}

            fused = fusion(embeddings)
            fused.sum().backward()

            for param in fusion.parameters():
                assert param.grad is not None
