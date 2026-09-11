import type { PredictionResponse } from "../types";

/** A realistic `backend === "live"` /predict response, for tests to override piecemeal. */
export function livePrediction(overrides: Partial<PredictionResponse> = {}): PredictionResponse {
  return {
    risk_score: 0.42,
    diagnosis: "Normal",
    reason: "Risk score 42.0% is 8.0 percentage points below the 50% detection threshold.",
    gradcam_heatmap: "",
    backend: "live",
    latency_ms: 12.3,
    quantum_expectations: [-0.31, 0.05, 0.77, -0.62],
    n_qubits: 4,
    quantum_backend: "default.qubit",
    base_image: "AXIAL_SLICE_12_BASE64",
    gradcam_overlay: "AXIAL_SLICE_12_OVERLAY_BASE64",
    gradcam_plane: "axial",
    gradcam_slice_index: 12,
    planes: {
      axial: {
        available: true,
        num_slices: 24,
        slices: Array.from({ length: 24 }, (_, i) => `AXIAL_SLICE_${i}_BASE64`),
      },
      coronal: { available: false, num_slices: 0, slices: [] },
      sagittal: { available: false, num_slices: 0, slices: [] },
    },
    primary_plane: "axial",
    primary_slice_index: 12,
    provenance: "live",
    provenance_label: "LIVE",
    model_source: "trained_checkpoint",
    model_source_label: "TRAINED MODEL",
    quantum_execution: "quantum_simulator",
    quantum_execution_label: "QUANTUM SIMULATOR",
    gradcam_degenerate: false,
    severity_band_normal_max: 0.4,
    severity_band_urgent_min: 0.75,
    ...overrides,
  };
}
