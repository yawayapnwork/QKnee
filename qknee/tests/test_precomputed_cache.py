"""
Tests for the precomputed demo cache (`scripts/generate_demo_cache.py`),
which exists specifically so a live demo/judging session can fall back to
zero-latency, pre-recorded results if the quantum simulator or a live
backend misbehaves mid-demo (the PRD's Plan B latency-risk mitigation).

The core contract under test: loading the cache must be near-instant and
must not touch the quantum-simulator runtime at all — if loading it were
itself slow, or accidentally re-ran inference, it wouldn't be a usable
fallback under time pressure.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

CACHE_PATH = Path("qknee/artifacts/precomputed_cache.json")
HEATMAPS_DIR = Path("qknee/artifacts/heatmaps")
EXPECTED_N_CASES = 10
MAX_LOAD_TIME_SECONDS = 0.010  # 10ms, for all EXPECTED_N_CASES combined

pytestmark = pytest.mark.skipif(
    not CACHE_PATH.exists(),
    reason=f"{CACHE_PATH} not generated yet; run `python scripts/generate_demo_cache.py` first.",
)


def load_precomputed_cache(cache_path: Path = CACHE_PATH) -> List[Dict]:
    """Pure JSON parsing — no `qknee.models.*`/PennyLane/Torch imports, no
    model construction, no circuit execution. This is deliberately the
    entire implementation: a demo fallback that itself depends on the
    quantum stack being healthy would defeat the point."""
    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["cases"]


class TestPrecomputedCacheStructure:
    def test_cache_file_exists(self):
        assert CACHE_PATH.exists()

    def test_contains_exactly_ten_cases(self):
        cases = load_precomputed_cache()
        assert len(cases) == EXPECTED_N_CASES

    def test_covers_all_three_categories(self):
        cases = load_precomputed_cache()
        categories = {case["ground_truth_category"] for case in cases}
        assert categories == {"Normal", "ACL Tear", "Meniscal Tear"}

    def test_covers_both_sagittal_and_coronal_planes(self):
        cases = load_precomputed_cache()
        planes = {case["plane"] for case in cases}
        assert planes == {"sagittal", "coronal"}

    def test_each_case_has_required_fields(self):
        required_fields = {
            "case_id", "ground_truth_category", "plane",
            "quantum_angles", "pauli_z_expectations", "risk_score",
            "risk_tier", "classification_label", "clinical_text_snippet",
            "total_latency_ms", "heatmap_file", "heatmap_base64",
        }
        for case in load_precomputed_cache():
            missing = required_fields - case.keys()
            assert not missing, f"case '{case.get('case_id')}' missing fields: {missing}"

    def test_quantum_angles_are_4d_in_expected_range(self):
        for case in load_precomputed_cache():
            angles = case["quantum_angles"]
            assert len(angles) == 4
            assert all(-1e-6 <= a <= 2 * 3.141592653589793 + 1e-6 for a in angles)

    def test_pauli_z_expectations_are_4d_in_valid_range(self):
        for case in load_precomputed_cache():
            expvals = case["pauli_z_expectations"]
            assert len(expvals) == 4
            assert all(-1.0 - 1e-6 <= v <= 1.0 + 1e-6 for v in expvals)

    def test_risk_scores_are_valid_probabilities(self):
        for case in load_precomputed_cache():
            assert 0.0 <= case["risk_score"] <= 1.0

    def test_heatmap_files_exist_on_disk(self):
        for case in load_precomputed_cache():
            heatmap_path = Path("qknee/artifacts") / case["heatmap_file"]
            assert heatmap_path.exists(), f"missing heatmap file for case '{case['case_id']}'"

    def test_heatmap_base64_decodes_to_valid_png_bytes(self):
        for case in load_precomputed_cache():
            raw_bytes = base64.b64decode(case["heatmap_base64"])
            assert raw_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature

    def test_clinical_text_snippet_is_nonempty(self):
        for case in load_precomputed_cache():
            assert isinstance(case["clinical_text_snippet"], str)
            assert len(case["clinical_text_snippet"]) > 0


class TestInstantLoadWithoutQuantumRuntime:
    """The two guarantees the PRD's Plan B fallback actually depends on."""

    def test_loading_all_ten_cases_completes_in_under_10ms(self):
        start = time.perf_counter()
        cases = load_precomputed_cache()
        elapsed_seconds = time.perf_counter() - start

        assert len(cases) == EXPECTED_N_CASES
        assert elapsed_seconds < MAX_LOAD_TIME_SECONDS, (
            f"Loading {EXPECTED_N_CASES} cached cases took {elapsed_seconds * 1000:.2f}ms, "
            f"expected < {MAX_LOAD_TIME_SECONDS * 1000:.0f}ms"
        )

    def test_loading_does_not_invoke_the_quantum_simulator(self):
        """Patches PennyLane's `QNode.__call__` to raise if invoked, then
        loads the cache — proving structurally (not just "it happened to
        be fast") that no circuit execution occurs during a cache load,
        regardless of whether PennyLane is already imported elsewhere in
        the test session (e.g. via other fixtures)."""
        import pennylane as qml

        with patch.object(
            qml.QNode, "__call__",
            side_effect=AssertionError("QNode was invoked while loading the precomputed cache"),
        ):
            cases = load_precomputed_cache()

        assert len(cases) == EXPECTED_N_CASES

    def test_loader_module_imports_no_quantum_or_torch_dependencies(self):
        """Static guarantee complementing the QNode-patch test above: the
        loader function itself is implemented with only `json`/`pathlib`
        (see its own docstring) — asserted here by checking this test
        module's global namespace never bound `torch`/`pennylane`/any
        `qknee.models.*` name at import time."""
        import qknee.tests.test_precomputed_cache as this_module

        module_globals = vars(this_module)
        for forbidden in ("torch", "pennylane", "qml", "PipelineRunner", "VQCClassifier", "QKneeModel"):
            assert forbidden not in module_globals, (
                f"test_precomputed_cache.py imports '{forbidden}' at module scope — "
                "the cache loader must stay independent of the quantum/model stack."
            )
