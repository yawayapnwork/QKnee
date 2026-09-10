import type { SeverityTag } from "./types";

/** Maps a `risk_score` (from either a live `/predict` response or a real,
 * offline-scored `/api/cases/{id}` response — both the same `PredictionResponse`
 * shape) to a display severity tier. Pure function of the score alone; carries
 * no notion of "which case" or "live vs. demo" — see `lib/provenance.ts` for that. */
export function severityFromRisk(riskScore: number): SeverityTag {
  if (riskScore >= 0.75) return "Urgent Surgical Consult";
  if (riskScore >= 0.4) return "Indeterminate";
  return "Normal";
}
