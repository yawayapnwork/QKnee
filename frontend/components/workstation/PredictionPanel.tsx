import { Gauge } from "@/components/ui/Gauge";
import { formatLatency } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

/**
 * The visual anchor of the analysis column: one dominant number, the
 * diagnosis inline with it, latency demoted to a caption. No competing
 * equal-weight metric cards (the old `TriageCard` gave "Diagnosis" and
 * "Latency" the same visual weight as the risk gauge itself).
 */
export function PredictionPanel({ result }: { result: DiagnosticResult }) {
  return (
    <div className="flex flex-col items-center gap-3 py-2 text-center">
      <h2 className="sr-only">Model result</h2>
      <Gauge value={result.riskScore} severity={result.severity} />
      <p className="text-lg font-semibold text-ink-primary">{result.diagnosis}</p>
      <p className="font-mono text-xs text-ink-faint">Inference latency: {formatLatency(result.latencyMs)}</p>
    </div>
  );
}
