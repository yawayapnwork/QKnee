"""
Performance benchmark: end-to-end pipeline inference latency per MRI slice.

These tests report timing (printed with `-s`) and assert only a generous
upper bound, so they catch a real performance regression (e.g. an
accidental O(n^2) loop, a forgotten `.cpu()`/`.numpy()` round-trip added to
a hot path) without being flaky on slower CI hardware. They are marked
`benchmark` (not a hard perf gate) and `slow` (exercises the real
ResNet18 + PennyLane stack).
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

# Generous ceiling for a single CPU forward pass through ResNet18 + PCA +
# a 4-qubit PennyLane state-vector simulation. Real hardware is expected to
# land well under this; the bound exists to catch regressions, not to
# certify production SLAs.
MAX_ACCEPTABLE_LATENCY_MS = 3000.0
N_WARMUP_RUNS = 2
N_TIMED_RUNS = 5


def _time_single_slice_inference(qknee_model, image: torch.Tensor) -> float:
    start = time.perf_counter()
    with torch.no_grad():
        qknee_model(image)
    return (time.perf_counter() - start) * 1000.0


class TestEndToEndLatency:
    def test_single_slice_latency_under_ceiling(self, qknee_model):
        single_slice = torch.rand(1, 3, 224, 224)

        for _ in range(N_WARMUP_RUNS):
            _time_single_slice_inference(qknee_model, single_slice)

        timings_ms = [
            _time_single_slice_inference(qknee_model, single_slice)
            for _ in range(N_TIMED_RUNS)
        ]

        median_ms = statistics.median(timings_ms)
        print(
            f"\n[latency] per-slice inference: "
            f"median={median_ms:.1f}ms, min={min(timings_ms):.1f}ms, "
            f"max={max(timings_ms):.1f}ms over {N_TIMED_RUNS} runs"
        )

        assert median_ms < MAX_ACCEPTABLE_LATENCY_MS, (
            f"Median per-slice latency {median_ms:.1f}ms exceeds the "
            f"{MAX_ACCEPTABLE_LATENCY_MS}ms regression ceiling"
        )

    def test_batched_inference_is_not_dramatically_slower_per_item(self, qknee_model):
        """Batch-of-8 throughput sanity check: per-item cost in a batch
        should not be drastically worse than single-item latency (would
        indicate e.g. an accidental per-sample Python loop somewhere in the
        forward path instead of vectorized batch ops)."""
        single_slice = torch.rand(1, 3, 224, 224)
        batch = torch.rand(8, 3, 224, 224)

        for _ in range(N_WARMUP_RUNS):
            _time_single_slice_inference(qknee_model, single_slice)
            _time_single_slice_inference(qknee_model, batch)

        single_ms = statistics.median(
            _time_single_slice_inference(qknee_model, single_slice) for _ in range(N_TIMED_RUNS)
        )
        batch_ms = statistics.median(
            _time_single_slice_inference(qknee_model, batch) for _ in range(N_TIMED_RUNS)
        )
        per_item_in_batch_ms = batch_ms / 8

        print(
            f"\n[latency] single={single_ms:.1f}ms, "
            f"batch-of-8={batch_ms:.1f}ms ({per_item_in_batch_ms:.1f}ms/item)"
        )

        # Allow generous headroom (batched item cost within 3x single-item
        # cost) since PennyLane's default.qubit simulator here loops over
        # the batch per-sample internally; this just guards against a
        # severe, unexpected blowup.
        assert per_item_in_batch_ms < single_ms * 3 + 50
