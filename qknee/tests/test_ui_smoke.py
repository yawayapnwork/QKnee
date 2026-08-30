"""
Smoke tests for `qknee.ui.analysis_app` and `qknee.ui.dashboard` (Streamlit
apps). These modules aren't run inside a Streamlit `ScriptRunContext` here,
so coverage is limited to:

    1. The modules import cleanly (no top-level `st.*` calls that require a
       running app, config loads, etc).
    2. Pure helper functions (image normalization, mock inference, slice
       extraction, URL resolution) behave correctly in isolation, without
       touching `st.*`.

UI rendering itself (`main()`, `render_header()`, `render_sidebar()`, ...)
is not exercised here — that requires a live Streamlit session and is out
of scope for pytest-level coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("streamlit")

import qknee.ui.analysis_app as analysis_app
import qknee.ui.dashboard as dashboard


# --------------------------------------------------------------------------- #
# 1. Import / module smoke
# --------------------------------------------------------------------------- #

class TestModuleImports:
    def test_analysis_app_exposes_expected_api(self):
        assert hasattr(analysis_app, "main")
        assert hasattr(analysis_app, "AnalysisResult")
        assert callable(analysis_app.run_mock_analysis)

    def test_dashboard_exposes_expected_api(self):
        assert hasattr(dashboard, "main")
        assert hasattr(dashboard, "InferenceResult")
        assert callable(dashboard.run_mock_inference)


# --------------------------------------------------------------------------- #
# 2a. analysis_app.py pure helpers
# --------------------------------------------------------------------------- #

class TestAnalysisAppHelpers:
    def test_normalize_uint8_maps_to_full_range(self):
        slice_2d = np.array([[0, 50], [100, 200]], dtype=np.float32)
        normalized = analysis_app.normalize_uint8(slice_2d)

        assert normalized.dtype == np.uint8
        assert normalized.min() == 0
        assert normalized.max() == 255

    def test_normalize_uint8_constant_slice_returns_zeros(self):
        slice_2d = np.full((4, 4), 42, dtype=np.float32)
        normalized = analysis_app.normalize_uint8(slice_2d)
        assert np.all(normalized == 0)

    def test_apply_contrast_identity_at_one(self):
        slice_uint8 = np.array([[0, 127, 255]], dtype=np.uint8)
        result = analysis_app.apply_contrast(slice_uint8, contrast=1.0)
        np.testing.assert_allclose(result, slice_uint8, atol=1)

    def test_apply_contrast_clips_to_valid_range(self):
        slice_uint8 = np.array([[0, 255]], dtype=np.uint8)
        result = analysis_app.apply_contrast(slice_uint8, contrast=5.0)
        assert result.min() >= 0 and result.max() <= 255

    def test_run_mock_analysis_is_deterministic_for_same_slice(self):
        rng = np.random.default_rng(0)
        slice_2d = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)

        result_a = analysis_app.run_mock_analysis(slice_2d)
        result_b = analysis_app.run_mock_analysis(slice_2d)

        assert result_a.risk_score == pytest.approx(result_b.risk_score)
        assert result_a.backend == "mock"
        assert 0.0 <= result_a.risk_score <= 1.0
        assert result_a.prediction_label in {"Normal", "Abnormality Detected"}
        assert result_a.gradcam_overlay is not None
        assert result_a.gradcam_overlay.shape == (32, 32, 3)

    def test_run_mock_analysis_differs_for_different_slices(self):
        slice_a = np.random.default_rng(1).integers(0, 255, size=(16, 16), dtype=np.uint8)
        slice_b = np.random.default_rng(2).integers(0, 255, size=(16, 16), dtype=np.uint8)

        result_a = analysis_app.run_mock_analysis(slice_a)
        result_b = analysis_app.run_mock_analysis(slice_b)

        assert result_a.risk_score != pytest.approx(result_b.risk_score)

    def test_run_mock_analysis_includes_bounded_pauli_z_expectations(self):
        slice_2d = np.random.default_rng(3).integers(0, 255, size=(16, 16), dtype=np.uint8)
        result = analysis_app.run_mock_analysis(slice_2d)

        assert result.pauli_z_expectations is not None
        assert len(result.pauli_z_expectations) == analysis_app.N_QUBITS
        assert np.all(result.pauli_z_expectations >= -1.0) and np.all(result.pauli_z_expectations <= 1.0)


# --------------------------------------------------------------------------- #
# 2a-bis. analysis_app.py multi-slice (volumetric) Grad-CAM + quantum panel
# --------------------------------------------------------------------------- #

class TestAnalysisAppVolumetric:
    def test_subsampled_indices_returns_full_range_when_under_cap(self):
        assert analysis_app._subsampled_indices(5, max_slices=40) == [0, 1, 2, 3, 4]

    def test_subsampled_indices_caps_and_spans_full_range(self):
        indices = analysis_app._subsampled_indices(200, max_slices=20)
        assert len(indices) <= 20
        assert indices[0] == 0
        assert indices[-1] == 199
        assert indices == sorted(indices)

    def test_get_pauli_z_expectations_returns_none_without_quantum_layer(self):
        class _NoQuantumLayerModel:
            pass

        result = analysis_app.get_pauli_z_expectations(_NoQuantumLayerModel(), np.zeros((1, 4), dtype=np.float32))
        assert result is None

    def test_get_pauli_z_expectations_reads_real_vqc_quantum_layer(self):
        from qknee.models.vqc import VQCClassifier

        torch_module = pytest.importorskip("torch")
        torch_module.manual_seed(0)
        vqc = VQCClassifier(n_qubits=4, n_layers=1)

        angles = np.zeros((1, 4), dtype=np.float32)
        expvals = analysis_app.get_pauli_z_expectations(vqc, angles)

        assert expvals is not None
        assert expvals.shape == (4,)
        assert np.all(expvals >= -1.0 - 1e-6) and np.all(expvals <= 1.0 + 1e-6)

    def test_run_volumetric_gradcam_mock_ranks_top_k_by_magnitude(self):
        volume = np.random.default_rng(4).integers(0, 255, size=(12, 32, 32), dtype=np.uint8)
        result = analysis_app.run_volumetric_gradcam_mock(volume, "Axial", max_slices=40, top_k=3)

        assert result.plane == "Axial"
        assert result.backend == "mock"
        assert set(result.analyzed_indices) == set(result.heatmaps.keys())
        assert len(result.top_k_indices) == 3
        expected = sorted(result.magnitudes, key=lambda i: result.magnitudes[i], reverse=True)[:3]
        assert result.top_k_indices == expected
        for heatmap in result.heatmaps.values():
            assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6

    def test_run_volumetric_gradcam_mock_respects_max_slices(self):
        volume = np.random.default_rng(5).integers(0, 255, size=(100, 16, 16), dtype=np.uint8)
        result = analysis_app.run_volumetric_gradcam_mock(volume, "Axial", max_slices=10, top_k=3)
        assert len(result.analyzed_indices) <= 10

    def test_volumetric_result_overlay_for_analyzed_slice(self):
        volume = np.random.default_rng(6).integers(0, 255, size=(6, 32, 32), dtype=np.uint8)
        result = analysis_app.run_volumetric_gradcam_mock(volume, "Coronal", max_slices=40, top_k=2)
        display_slice = analysis_app.normalize_uint8(volume[0])

        from qknee.xai.gradcam import COLORMAP_OPTIONS

        overlay = result.overlay_for(0, display_slice, alpha=0.5, colormap=COLORMAP_OPTIONS["jet"])
        assert overlay is not None
        assert overlay.shape == (32, 32, 3)

    def test_volumetric_result_overlay_for_unanalyzed_slice_is_none(self):
        volume = np.random.default_rng(7).integers(0, 255, size=(6, 32, 32), dtype=np.uint8)
        result = analysis_app.run_volumetric_gradcam_mock(volume, "Coronal", max_slices=40, top_k=2)
        display_slice = analysis_app.normalize_uint8(volume[0])

        overlay = result.overlay_for(9999, display_slice, alpha=0.5, colormap=6)
        assert overlay is None

    def test_render_quantum_attribution_panel_none_when_no_expectations(self):
        assert analysis_app.render_quantum_attribution_panel(None) is None

    def test_render_quantum_attribution_panel_returns_figure(self):
        expvals = np.array([-0.8, -0.1, 0.3, 0.9], dtype=np.float32)
        fig = analysis_app.render_quantum_attribution_panel(expvals)
        assert fig is not None


# --------------------------------------------------------------------------- #
# 2b. dashboard.py pure helpers
# --------------------------------------------------------------------------- #

class TestDashboardHelpers:
    def test_normalize_for_display_maps_to_full_range(self):
        slice_2d = np.array([[0, 50], [100, 200]], dtype=np.float32)
        normalized = dashboard.normalize_for_display(slice_2d)
        assert normalized.dtype == np.uint8
        assert normalized.min() == 0
        assert normalized.max() == 255

    def test_get_slice_axial_coronal_sagittal(self):
        volume = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        axial = dashboard.get_slice(volume, "Axial", 0)
        coronal = dashboard.get_slice(volume, "Coronal", 1)
        sagittal = dashboard.get_slice(volume, "Sagittal", 2)

        np.testing.assert_array_equal(axial, volume[0, :, :])
        np.testing.assert_array_equal(coronal, volume[:, 1, :])
        np.testing.assert_array_equal(sagittal, volume[:, :, 2])

    def test_get_slice_unknown_view_raises(self):
        volume = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="Unknown view"):
            dashboard.get_slice(volume, "Oblique", 0)

    def test_view_axis_size_matches_volume_shape(self):
        volume = np.zeros((5, 6, 7))
        assert dashboard.view_axis_size(volume, "Axial") == 5
        assert dashboard.view_axis_size(volume, "Coronal") == 6
        assert dashboard.view_axis_size(volume, "Sagittal") == 7

    def test_resolve_api_url_reads_env_var(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("QKNEE_API_URL", raising=False)
        assert dashboard.resolve_api_url() is None

        monkeypatch.setenv("QKNEE_API_URL", "http://example.internal:8000")
        assert dashboard.resolve_api_url() == "http://example.internal:8000"

    def test_api_is_reachable_returns_false_on_connection_error(self):
        # Port 1 is reserved/unlikely to have a listener; a fast, well-defined failure.
        assert dashboard.api_is_reachable("http://127.0.0.1:1", timeout=0.2) is False

    def test_run_mock_inference_is_deterministic_and_bounded(self):
        rng = np.random.default_rng(0)
        slice_2d = rng.integers(0, 255, size=(32, 32), dtype=np.uint8)

        result_a = dashboard.run_mock_inference(slice_2d)
        result_b = dashboard.run_mock_inference(slice_2d)

        assert result_a.acl_risk == pytest.approx(result_b.acl_risk)
        assert result_a.meniscus_risk == pytest.approx(result_b.meniscus_risk)
        assert result_a.backend == "mock"
        assert 0.0 <= result_a.acl_risk <= 1.0
        assert 0.0 <= result_a.meniscus_risk <= 1.0
        assert result_a.total_latency_ms == pytest.approx(
            result_a.resnet_latency_ms + result_a.pca_latency_ms + result_a.quantum_latency_ms
        )
