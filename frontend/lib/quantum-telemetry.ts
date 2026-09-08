import type { PredictionResponse, PresetCase, QuantumTelemetry } from "./types";

const UNAVAILABLE: QuantumTelemetry = {
  provenance: "unavailable",
  expectations: [],
  nQubits: null,
  device: null,
};

/**
 * Derives this result's `QuantumTelemetry` **only** from the backend's own
 * `/predict` response — never from a preset case, never from `risk_score`,
 * never randomly generated. This is the single call site the live-inference
 * path is allowed to use; it deliberately takes no `PresetCase`/`activeCase`
 * argument at all, so preset telemetry cannot leak into a live result even
 * by accident (see `__tests__/quantum-telemetry.test.ts`).
 *
 * `prediction.backend === "live"`: this exact upload's own executed VQC
 * circuit — labeled "live".
 * `prediction.backend` starting with "cache-fallback/": real Pauli-Z
 * measurements, but recorded ahead of time for a cached demo case, not this
 * upload — labeled "precomputed-demo".
 * Anything else (e.g. `backend === "mock"`, or no `quantum_expectations` in
 * the response at all): "unavailable" — the UI must say so, not fabricate a
 * value.
 */
export function quantumTelemetryFromPrediction(prediction: PredictionResponse): QuantumTelemetry {
  const expectations = prediction.quantum_expectations;
  if (!expectations || expectations.length === 0) {
    return UNAVAILABLE;
  }

  if (prediction.backend === "live") {
    return {
      provenance: "live",
      expectations,
      nQubits: prediction.n_qubits ?? expectations.length,
      device: prediction.quantum_backend ?? null,
    };
  }

  if (prediction.backend.startsWith("cache-fallback/")) {
    return {
      provenance: "precomputed-demo",
      expectations,
      nQubits: prediction.n_qubits ?? expectations.length,
      device: prediction.quantum_backend ?? null,
    };
  }

  return UNAVAILABLE;
}

/** Builds a preset case's demo telemetry — always "precomputed-demo", never "live". */
export function quantumTelemetryFromPreset(preset: PresetCase): QuantumTelemetry {
  return {
    provenance: "precomputed-demo",
    expectations: preset.qubitExpectations,
    nQubits: preset.qubitExpectations.length,
    device: null,
  };
}
