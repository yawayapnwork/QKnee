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
  // Deliberately NOT styled or worded like the two labels above anywhere it
  // renders (see `QuantumTelemetry.tsx`) -- this is not a third execution
  // outcome the frontend can attest to, it is an explicit admission that it
  // can't. Never shortened to "DEMO" or similar; a viewer must not be able
  // to mentally file it under either "ran" or "didn't run".
  not_verifiable: "NOT INDEPENDENTLY VERIFIABLE",
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

// `provenanceForPreset` (hand-authored fake-case provenance) is deleted --
// demo cases are now real RSNA Knee studies scored once offline (see
// `scripts/build_real_demo_cases.py`) and served via `GET /api/cases/{id}`
// as an ordinary `PredictionResponse`, so they go through
// `provenanceFromPrediction` above like any other result. Their
// `provenance` comes back `"precomputed_demo"` from the backend itself
// (real model, real quantum circuit, computed offline and replayed), not
// from a frontend-side constructor.

