import { provenanceForFailedLiveFallback, provenanceForPreset } from "./provenance";
import { quantumTelemetryFromPreset } from "./quantum-telemetry";
import { volumeViewFromPreset } from "./viewer";
import type { DiagnosticResult, PresetCase, SeverityTag } from "./types";

export const PRESET_CASES: PresetCase[] = [
  {
    id: "sample-01",
    label: "Sample 01",
    description: "Confirmed ACL Tear",
    category: "ACL Tear",
    riskScore: 0.912,
    qubitExpectations: [-0.71, 0.62, -0.48, 0.83],
  },
  {
    id: "sample-02",
    label: "Sample 02",
    description: "Intact Meniscus",
    category: "Intact Meniscus",
    riskScore: 0.081,
    qubitExpectations: [0.12, -0.05, 0.09, -0.14],
  },
  {
    id: "sample-03",
    label: "Sample 03",
    description: "Complex Multi-Compartment Defect",
    category: "Multi-Compartment Defect",
    riskScore: 0.584,
    qubitExpectations: [-0.01, 0.19, -0.05, -0.08],
  },
];

export function severityFromRisk(riskScore: number): SeverityTag {
  if (riskScore >= 0.75) return "Urgent Surgical Consult";
  if (riskScore >= 0.4) return "Indeterminate";
  return "Normal";
}

/**
 * Builds a preset/demo case's `DiagnosticResult`.
 *
 * `isFailedLiveFallback` distinguishes two very different situations that
 * both happen to reuse the same canned preset data (AUDIT.md P1 #7
 * requirements 6/8):
 *   - `false` (default): the viewer deliberately picked this preset case
 *     from the case switcher — provenance is "precomputed_demo", an
 *     explicitly-marked, non-alarming demo state.
 *   - `true`: a live upload's own inference call just failed and this
 *     preset is standing in for it — provenance is "mock_fallback", the
 *     loud/unmistakable state, so a viewer can never mistake a failed live
 *     call recovering to canned data for a routine demo selection.
 */
export function mockDiagnosticResult(preset: PresetCase, isFailedLiveFallback = false): DiagnosticResult {
  return {
    riskScore: preset.riskScore,
    diagnosis: preset.riskScore >= 0.5 ? "Tear Detected" : "Normal",
    severity: severityFromRisk(preset.riskScore),
    backend: isFailedLiveFallback ? "mock/cold-start-fallback" : "mock/preset",
    latencyMs: null,
    quantumTelemetry: quantumTelemetryFromPreset(preset),
    volume: volumeViewFromPreset(preset),
    source: "mock",
    provenance: isFailedLiveFallback ? provenanceForFailedLiveFallback() : provenanceForPreset(),
  };
}
