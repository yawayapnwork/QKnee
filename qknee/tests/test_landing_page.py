"""
Tests for `qknee.ui.landing_page` (the public entry view wired into
`qknee.ui.dashboard.main()`). Covers:

    1. Pure helper functions (dynamic metric computation, showcase-case
       heatmap decoding) in isolation, without a running Streamlit session
       — same convention as `test_ui_smoke.py`.
    2. Actual rendering, via Streamlit's `AppTest` harness (a real script
       run, not just an import-time smoke check): the landing page loads
       without exceptions, its CTA buttons navigate `dashboard.main()` to
       the right view/tab, and the live sample showcase's 1-click preview
       actually reveals a case's risk metric.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")

import qknee.ui.landing_page as landing_page

pytestmark = [pytest.mark.slow]


@pytest.fixture(autouse=True)
def _clear_cached_data():
    """`@st.cache_data`-decorated functions cache process-globally, keyed
    on their (here, argument-less) call signature — NOT on the module
    globals a test monkeypatches (e.g. `PRECOMPUTED_CACHE_PATH`). Without
    this, a test that monkeypatches a path and populates the cache would
    leak a stale/wrong result into every later test in this process
    (including the `AppTest` rendering tests below, which re-import and
    call these same cached functions against the real artifacts)."""
    landing_page._load_precomputed_cases.clear()
    landing_page._load_benchmark_results.clear()
    yield
    landing_page._load_precomputed_cases.clear()
    landing_page._load_benchmark_results.clear()


# --------------------------------------------------------------------------- #
# 1. Pure helpers
# --------------------------------------------------------------------------- #

class TestModuleImports:
    def test_exposes_expected_api(self):
        assert callable(landing_page.render_landing_page)
        assert callable(landing_page.render_hero)
        assert callable(landing_page.render_pipeline_explainer)
        assert callable(landing_page.render_live_sample_showcase)

    def test_view_state_constants_are_distinct(self):
        values = {landing_page.VIEW_LANDING, landing_page.VIEW_DIAGNOSTIC, landing_page.VIEW_BENCHMARK}
        assert len(values) == 3

    def test_tagline_matches_the_required_copy(self):
        assert landing_page.TAGLINE == (
            "NISQ-Ready Variational Quantum Machine Learning for Rapid Orthopedic Triage"
        )

    def test_showcase_covers_exactly_three_cases(self):
        assert len(landing_page.SHOWCASE_CASE_IDS) == 3


class TestParameterReductionMetric:
    def test_returns_a_percentage_between_zero_and_one_hundred(self):
        pct = landing_page._parameter_reduction_pct()
        assert 0.0 < pct < 100.0

    def test_matches_the_deck_asset_scripts_own_formula(self):
        """Cross-checks against `scripts/generate_deck_assets.py`'s
        independently-implemented parameter-count formulas — both must
        agree on the same architecture, since they describe the same
        model."""
        config = landing_page._config
        feature_dim = config.resnet.feature_dim
        n_qubits = config.quantum.n_qubits
        n_layers = config.quantum.n_layers

        linear_params = (feature_dim * n_qubits + n_qubits) + (n_qubits * 2 + 2)
        vqc_params = n_layers * n_qubits * 3 + n_qubits + 1
        expected_pct = (1 - vqc_params / linear_params) * 100

        assert landing_page._parameter_reduction_pct() == pytest.approx(expected_pct)


class TestQuantumLatencyMetric:
    def test_returns_none_when_no_benchmark_results_exist(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(landing_page, "BENCHMARK_RESULTS_PATH", tmp_path / "does_not_exist.json")
        landing_page._load_benchmark_results.clear()  # bust the @st.cache_data memo
        assert landing_page._quantum_latency_ms() is None

    def test_reads_the_vqc_models_measured_latency(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        results_path = tmp_path / "benchmark_results.json"
        results_path.write_text(json.dumps({
            "models": [
                {"name": "Classical Linear (ResNet18->4D Linear->Softmax)", "latency_ms_per_sample": 0.5},
                {"name": "Hybrid Q-Knee (ResNet18->PCA(4)->4-Qubit VQC)", "latency_ms_per_sample": 12.7},
            ]
        }), encoding="utf-8")
        monkeypatch.setattr(landing_page, "BENCHMARK_RESULTS_PATH", results_path)
        landing_page._load_benchmark_results.clear()

        assert landing_page._quantum_latency_ms() == pytest.approx(12.7)

    def test_returns_none_when_results_are_malformed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        results_path = tmp_path / "benchmark_results.json"
        results_path.write_text("not valid json{", encoding="utf-8")
        monkeypatch.setattr(landing_page, "BENCHMARK_RESULTS_PATH", results_path)
        landing_page._load_benchmark_results.clear()

        assert landing_page._quantum_latency_ms() is None


class TestDecodeCaseOverlay:
    def test_returns_none_when_no_heatmap_embedded(self):
        assert landing_page._decode_case_overlay({"case_id": "x"}) is None

    def test_decodes_a_valid_base64_png(self):
        import cv2

        raw_image = np.zeros((8, 8, 3), dtype=np.uint8)
        raw_image[..., 2] = 255  # pure red in BGR
        success, encoded = cv2.imencode(".png", raw_image)
        assert success
        case = {"case_id": "x", "heatmap_base64": base64.b64encode(encoded.tobytes()).decode("ascii")}

        decoded = landing_page._decode_case_overlay(case)
        assert decoded is not None
        assert decoded.shape == (8, 8, 3)
        np.testing.assert_array_equal(decoded, raw_image)

    def test_returns_none_for_corrupted_base64_without_raising(self):
        case = {"case_id": "x", "heatmap_base64": "not-valid-base64-png-data!!"}
        assert landing_page._decode_case_overlay(case) is None


class TestLoadPrecomputedCases:
    def test_returns_empty_list_when_cache_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(landing_page, "PRECOMPUTED_CACHE_PATH", tmp_path / "does_not_exist.json")
        landing_page._load_precomputed_cases.clear()
        assert landing_page._load_precomputed_cases() == []

    def test_loads_cases_from_a_real_cache_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cache_path = tmp_path / "precomputed_cache.json"
        cache_path.write_text(json.dumps({"cases": [{"case_id": "case_0001"}]}), encoding="utf-8")
        monkeypatch.setattr(landing_page, "PRECOMPUTED_CACHE_PATH", cache_path)
        landing_page._load_precomputed_cases.clear()

        cases = landing_page._load_precomputed_cases()
        assert [c["case_id"] for c in cases] == ["case_0001"]


# --------------------------------------------------------------------------- #
# 2. Actual rendering via Streamlit's AppTest harness
# --------------------------------------------------------------------------- #

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
_DASHBOARD_PATH = str(Path(__file__).resolve().parents[2] / "qknee" / "ui" / "dashboard.py")


class TestLandingPageRendering:
    def test_dashboard_lands_on_the_landing_page_by_default(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()

        assert not at.exception
        button_labels = [b.label for b in at.button]
        assert "🚀 Launch Live Diagnostic Console" in button_labels
        assert "📊 Explore Clinical Benchmarks" in button_labels

    def test_hero_renders_three_dynamic_metric_cards(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()

        assert not at.exception
        metric_labels = {m.label for m in at.metric}
        assert {"Parameter Reduction", "Quantum Circuit Latency", "Variational Circuit"} <= metric_labels

    def test_pipeline_explainer_renders_three_tabs(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()

        assert not at.exception
        tab_labels = [t.label for t in at.tabs]
        assert tab_labels == [
            "① Spatial Vision Backbone",
            "② Quantum Circuit Execution",
            "③ Explainability & Report",
        ]

    def test_launch_diagnostic_cta_routes_to_login_when_unauthenticated(self):
        """As of `qknee.ui.auth_view`'s integration, the Diagnostic
        workspace requires authentication — an unauthenticated visitor
        clicking this CTA lands on the login page (with `qknee_active_view`
        left as "diagnostic" so they land on that exact tab once they do
        sign in; see `test_demo_login_after_launch_cta_lands_on_diagnostic_tab`)."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        launch_button = next(b for b in at.button if "Launch Live Diagnostic" in b.label)

        launch_button.click().run()

        assert not at.exception
        assert at.session_state["qknee_active_view"] == "diagnostic"
        assert at.session_state["current_page"] == "login"
        assert [t.label for t in at.tabs] == ["🔐 Sign In", "📝 Create Account"]

    def test_demo_login_after_launch_cta_lands_on_diagnostic_tab(self):
        """The full unauthenticated-CTA-then-sign-in round trip: the
        visitor's originally-requested "diagnostic" destination survives
        the login detour."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Launch Live Diagnostic" in b.label).click().run()
        demo_button = next(b for b in at.button if "Demo Account" in b.label)

        demo_button.click().run()

        assert not at.exception
        assert at.session_state["authenticated"] is True
        assert at.session_state["current_page"] == "workspace"
        assert at.tabs[0].label == "🔬 Diagnostic View"

    def test_explore_benchmarks_cta_switches_to_benchmark_tab_first(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        benchmarks_button = next(b for b in at.button if "Explore Clinical Benchmarks" in b.label)

        benchmarks_button.click().run()

        assert not at.exception
        assert at.session_state["qknee_active_view"] == "benchmark"
        assert at.tabs[0].label == "📊 Quantum vs Classical Benchmark"

    def test_back_to_home_returns_to_the_landing_view(self):
        """The sidebar 'Back to Home' button only renders once inside the
        workspace, which (for the Diagnostic tab) now requires signing in
        first — so this test signs in via the Demo Account before
        exercising it."""
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        next(b for b in at.button if "Launch Live Diagnostic" in b.label).click().run()
        next(b for b in at.button if "Demo Account" in b.label).click().run()
        home_button = next(b for b in at.sidebar.button if "Back to Home" in b.label)

        home_button.click().run()

        assert not at.exception
        assert at.session_state["qknee_active_view"] == "landing"
        assert at.session_state["current_page"] == "landing"
        assert any("Launch Live Diagnostic" in b.label for b in at.button)

    def test_showcase_preview_click_reveals_the_case_risk_metric(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()
        preview_button = next(b for b in at.button if "Preview this case" in b.label)

        preview_button.click().run()

        assert not at.exception
        assert any(m.label == "Tear Risk" for m in at.metric)

    def test_showcase_shows_one_preview_button_per_case(self):
        at = AppTest.from_file(_DASHBOARD_PATH, default_timeout=60)
        at.run()

        assert not at.exception
        preview_buttons = [b for b in at.button if "Preview this case" in b.label]
        assert len(preview_buttons) == len(landing_page.SHOWCASE_CASE_IDS)
