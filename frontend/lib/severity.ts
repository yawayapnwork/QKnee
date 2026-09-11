import type { SeverityTag } from "./types";

export interface SeverityThresholds {
  normalMax: number;
  urgentMin: number;
}

/** Maps a `risk_score` (from either a live `/predict` response or a real,
 * offline-scored `/api/cases/{id}` response — both the same `PredictionResponse`
 * shape) to a display severity tier, using the band boundaries the backend
 * computed `risk_score` against (`PredictionResponse.severity_band_normal_max`/
 * `severity_band_urgent_min`) — never a value hardcoded here. Carries no notion
 * of "which case" or "live vs. demo" — see `lib/provenance.ts` for that. */
export function severityFromRisk(riskScore: number, thresholds: SeverityThresholds): SeverityTag {
  if (riskScore >= thresholds.urgentMin) return "Urgent Surgical Consult";
  if (riskScore >= thresholds.normalMax) return "Indeterminate";
  return "Normal";
}

/** The `[lower, upper)` `risk_score` bounds (as fractions in [0, 1]) of `severity`'s
 * band, given the same backend-provided `thresholds` `severityFromRisk` used to
 * compute it. `Infinity` upper bound for "Urgent Surgical Consult" since that band
 * is open-ended above `urgentMin`. */
export function severityBounds(severity: SeverityTag, thresholds: SeverityThresholds): [number, number] {
  switch (severity) {
    case "Normal":
      return [0, thresholds.normalMax];
    case "Indeterminate":
      return [thresholds.normalMax, thresholds.urgentMin];
    case "Urgent Surgical Consult":
      return [thresholds.urgentMin, 1];
  }
}
