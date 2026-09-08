import type { ModelSource, PredictionResponse, ProvenanceInfo } from "./types";

/**
 * Unified prediction-provenance vocabulary for the Next.js workstation —
 * the frontend counterpart of `qknee.observability.provenance` (the shared
 * Python module `extras/api/server.py` and `qknee/ui/dashboard.py` both
 * derive their own badges from), so the same underlying state is described
 * with exactly the same words on every surface.
 *
 * Fixes AUDIT.md P1 #5 (D2 — no prominent mock/live indicator) and P1 #7
 * (B3/C4b — a checkpoint fallback silently degrading every score; C4c —
 * "SIMULATION MODE" ambiguity).
 */

export const PROVENANCE_LABELS = {
  live: "LIVE",
  precomputed_demo: "PRECOMPUTED DEMO",
  mock_fallback: "MOCK/FALLBACK",
  cached: "CACHED",
  proxy: "PROXY",
} as const;

export const MODEL_SOURCE_LABELS: Record<ModelSource, string> = {
  trained_checkpoint: "TRAINED MODEL",
  random_fallback: "MODEL FALLBACK (untrained weights)",
};

export const QUANTUM_EXECUTION_LABELS = {
  quantum_simulator: "QUANTUM SIMULATOR",
  unavailable: "QUANTUM TELEMETRY UNAVAILABLE",
} as const;

/**
 * Builds this result's `ProvenanceInfo` **only** from the backend's own
 * `/predict` response — mirrors `quantum-telemetry.ts`'s single-call-site
 * pattern: the API already computed the authoritative classification (see
 * `qknee.observability.provenance.classify`), so this function's only job
 * is to reshape the wire (snake_case) fields into the app-level
 * (camelCase) shape, never to re-derive or override them from anything
 * else (e.g. `risk_score`, an `activeCase`/preset).
 */
export function provenanceFromPrediction(prediction: PredictionResponse): ProvenanceInfo {
  return {
    provenance: prediction.provenance,
    provenanceLabel: prediction.provenance_label,
    modelSource: prediction.model_source,
    modelSourceLabel: prediction.model_source_label,
    quantumExecution: prediction.quantum_execution,
    quantumExecutionLabel: prediction.quantum_execution_label,
    isTrustworthy: prediction.provenance !== "mock_fallback",
  };
}

/**
 * Builds a preset/demo case's provenance — always "precomputed_demo"
 * (AUDIT.md P1 #7 requirement 8: preset cases may remain usable for the
 * demo, but must be explicitly marked as such, never presented as "live").
 */
export function provenanceForPreset(): ProvenanceInfo {
  return {
    provenance: "precomputed_demo",
    provenanceLabel: PROVENANCE_LABELS.precomputed_demo,
    modelSource: null,
    modelSourceLabel: null,
    // Preset cases carry real, precomputed per-qubit values (see
    // PresetCase.qubitExpectations / quantumTelemetryFromPreset) -- not
    // random placeholders -- so a real circuit execution genuinely
    // happened at some point, just not for this request. Marking this
    // "unavailable" would contradict the non-empty telemetry the panel
    // displays right next to this badge.
    quantumExecution: "quantum_simulator",
    quantumExecutionLabel: QUANTUM_EXECUTION_LABELS.quantum_simulator,
    isTrustworthy: true,
  };
}


