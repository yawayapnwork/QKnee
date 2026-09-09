import type { PredictionResponse, PresetCase, QuantumTelemetry } from "./types";

const UNAVAILABLE: QuantumTelemetry = { expectations: [], nQubits: null, device: null };

/**
 * Extracts this result's raw per-qubit circuit output **only** from the
 * backend's own `/predict` response — never from a preset case, never
 * derived from `risk_score`, never randomly generated. This function does
 * NOT decide whether the data is trustworthy/live/demo — that question has
 * exactly one answer, computed once by `provenanceFromPrediction`
 * (`lib/provenance.ts`) and read from `DiagnosticResult.provenance`. This
 * function only ever answers "what are the numbers," never "what do the
 * numbers mean."
 */
export function quantumTelemetryFromPrediction(prediction: PredictionResponse): QuantumTelemetry {
  const expectations = prediction.quantum_expectations;
  if (!expectations || expectations.length === 0) {
    return UNAVAILABLE;
  }
  return {
    expectations,
    nQubits: prediction.n_qubits ?? expectations.length,
    device: prediction.quantum_backend ?? null,
  };
}

/** A preset/demo case's real, precomputed per-qubit values. */
export function quantumTelemetryFromPreset(preset: PresetCase): QuantumTelemetry {
  return {
    expectations: preset.qubitExpectations,
    nQubits: preset.qubitExpectations.length,
    device: null,
  };
}

/**
 * The ONE authoritative answer to "is there quantum telemetry data to
 * render" -- used by both `QuantumTelemetry.tsx` (the panel) and
 * `ReportExport.tsx` (the markdown export) so they can never disagree
 * about whether a result's per-qubit numbers should be shown. This is
 * deliberately independent of whether the execution is verified
 * (`provenance.quantumExecution === "quantum_simulator"`) -- a precomputed
 * demo case has real, non-empty `expectations` and IS shown, just labeled
 * "NOT INDEPENDENTLY VERIFIABLE" rather than "QUANTUM SIMULATOR" (see
 * `lib/provenance.ts#provenanceForPreset`). Only a genuinely empty
 * `expectations` array (`quantumExecution === "unavailable"`) hides the
 * panel/table entirely.
 */
export function hasQuantumTelemetry(telemetry: QuantumTelemetry): boolean {
  return telemetry.expectations.length > 0;
}
