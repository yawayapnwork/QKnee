"""
Tests for `qknee.observability.provenance` -- the single shared module both
`extras/api/server.py` (Next.js/FastAPI contract) and `qknee/ui/dashboard.py`
(Streamlit) derive their prediction-provenance badges from.

Covers:
    1. The five top-level `Provenance` categories `classify()` can produce,
       from each backend tag this project actually emits.
    2. AUDIT.md C4b's exact fix: a "live" backend tag with
       `model_checkpoint_loaded=False` must be downgraded to
       `mock_fallback` provenance, never reported as `live`.
    3. `QuantumExecution` never uses the ambiguous "SIMULATION MODE" wording
       (AUDIT.md C4c) -- exactly two labels exist, and neither is that string.
    4. `ProvenanceInfo.is_trustworthy` / `.as_dict()`.
"""

from __future__ import annotations

import pytest

from qknee.observability.provenance import (
    MODEL_SOURCE_LABELS,
    PROVENANCE_LABELS,
    QUANTUM_EXECUTION_LABELS,
    ProvenanceInfo,
    classify,
)


class TestClassifyProvenanceCategory:
    def test_live_backend_with_a_trained_checkpoint_is_live(self):
        info = classify(backend_tag="live", model_checkpoint_loaded=True, quantum_expectations_present=True)
        assert info.provenance == "live"
        assert info.provenance_label == "LIVE"
        assert info.model_source == "trained_checkpoint"

    def test_mock_backend_is_mock_fallback_with_no_model_source(self):
        info = classify(backend_tag="mock", model_checkpoint_loaded=None, quantum_expectations_present=False)
        assert info.provenance == "mock_fallback"
        assert info.provenance_label == "MOCK/FALLBACK"
        assert info.model_source is None

    def test_cache_fallback_tag_is_precomputed_demo(self):
        info = classify(
            backend_tag="cache-fallback/case_0007", model_checkpoint_loaded=None, quantum_expectations_present=True,
        )
        assert info.provenance == "precomputed_demo"
        assert info.provenance_label == "PRECOMPUTED DEMO"

    def test_streamlit_cached_sidebar_tags_are_precomputed_demo(self):
        for tag in ("cached/case_1", "cached-fastpath/case_2"):
            info = classify(backend_tag=tag, model_checkpoint_loaded=None, quantum_expectations_present=True)
            assert info.provenance == "precomputed_demo", tag

    def test_streamlit_api_tag_is_proxy(self):
        info = classify(backend_tag="api/live", model_checkpoint_loaded=None, quantum_expectations_present=True)
        assert info.provenance == "proxy"
        assert info.provenance_label == "PROXY"

    def test_explicit_proxy_tag_is_proxy(self):
        info = classify(backend_tag="proxy", model_checkpoint_loaded=None, quantum_expectations_present=False)
        assert info.provenance == "proxy"

    def test_unrecognized_backend_tag_is_conservatively_mock_fallback_never_live(self):
        """An unknown/future backend tag must never be silently trusted as
        'live' just because it isn't recognized as one of the known
        degraded cases."""
        info = classify(backend_tag="some-future-tag", model_checkpoint_loaded=None, quantum_expectations_present=True)
        assert info.provenance == "mock_fallback"


class TestCheckpointFallbackDowngrade:
    """AUDIT.md C4b: the exact bug this module exists to fix."""

    def test_live_backend_tag_with_untrained_weights_is_downgraded_to_mock_fallback(self):
        info = classify(backend_tag="live", model_checkpoint_loaded=False, quantum_expectations_present=True)

        assert info.provenance == "mock_fallback"
        assert info.provenance_label == "MOCK/FALLBACK"
        assert info.model_source == "random_fallback"
        assert info.model_source_label == "MODEL FALLBACK (untrained weights)"
        assert info.is_trustworthy is False

    def test_live_backend_tag_with_trained_weights_stays_live_and_trustworthy(self):
        info = classify(backend_tag="live", model_checkpoint_loaded=True, quantum_expectations_present=True)
        assert info.provenance == "live"
        assert info.is_trustworthy is True


class TestQuantumExecutionNeverSaysSimulationMode:
    """AUDIT.md C4c: the quantum simulator running is this project's normal,
    non-degraded state and must never share a label with 'something is
    actually broken/missing'."""

    def test_only_two_quantum_execution_labels_exist(self):
        assert set(QUANTUM_EXECUTION_LABELS.values()) == {"QUANTUM SIMULATOR", "QUANTUM TELEMETRY UNAVAILABLE"}

    def test_no_label_anywhere_in_this_module_says_simulation_mode(self):
        all_labels = list(PROVENANCE_LABELS.values()) + list(MODEL_SOURCE_LABELS.values()) + list(
            QUANTUM_EXECUTION_LABELS.values()
        )
        assert not any("SIMULATION MODE" in label.upper() for label in all_labels)

    def test_real_quantum_expectations_present_means_quantum_simulator_executed(self):
        info = classify(backend_tag="live", model_checkpoint_loaded=True, quantum_expectations_present=True)
        assert info.quantum_execution == "quantum_simulator"
        assert info.quantum_execution_label == "QUANTUM SIMULATOR"

    def test_no_quantum_expectations_means_unavailable_not_simulation_mode(self):
        info = classify(backend_tag="mock", model_checkpoint_loaded=None, quantum_expectations_present=False)
        assert info.quantum_execution == "unavailable"
        assert info.quantum_execution_label == "QUANTUM TELEMETRY UNAVAILABLE"


class TestProvenanceInfoHelpers:
    def test_as_dict_round_trips_every_field(self):
        info = classify(backend_tag="live", model_checkpoint_loaded=True, quantum_expectations_present=True)
        as_dict = info.as_dict()
        assert as_dict == {
            "provenance": "live",
            "provenance_label": "LIVE",
            "model_source": "trained_checkpoint",
            "model_source_label": "TRAINED MODEL",
            "quantum_execution": "quantum_simulator",
            "quantum_execution_label": "QUANTUM SIMULATOR",
        }

    def test_model_source_label_is_none_exactly_when_model_source_is_none(self):
        info = ProvenanceInfo(provenance="mock_fallback", model_source=None, quantum_execution="unavailable")
        assert info.model_source_label is None

    @pytest.mark.parametrize("provenance", ["live", "precomputed_demo", "cached", "proxy"])
    def test_every_provenance_except_mock_fallback_is_trustworthy(self, provenance):
        info = ProvenanceInfo(provenance=provenance, model_source=None, quantum_execution="unavailable")
        assert info.is_trustworthy is True

    def test_mock_fallback_is_never_trustworthy(self):
        info = ProvenanceInfo(provenance="mock_fallback", model_source=None, quantum_execution="unavailable")
        assert info.is_trustworthy is False
