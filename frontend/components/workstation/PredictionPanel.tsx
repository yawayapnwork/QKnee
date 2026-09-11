import { Gauge } from "@/components/ui/Gauge";
import { ProvenanceBadge } from "@/components/workstation/ProvenanceBadge";
import { formatLatency, formatPercent } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

/**
 * RIGHT zone, top section: provenance first (never let a viewer read a
 * risk number before knowing whether it's LIVE/DEMO/MOCK), then the one
 * dominant number (the gauge), then a compact fact row. "Model status"
 * below is `provenance.modelSourceLabel` — the real, backend-reported
 * checkpoint state for THIS prediction ("TRAINED MODEL" /
 * "MODEL FALLBACK (untrained weights)"). There is deliberately no
 * separate "model version" row: the API exposes no semantic-version
 * field for the model, and inventing one to fill a slot would be exactly
 * the kind of fabricated fact this product's audit trail exists to
 * prevent — model status already carries the real, honest equivalent.
 */
export function PredictionPanel({ result }: { result: DiagnosticResult }) {
  return (
    <div className="flex flex-col gap-4">
      <ProvenanceBadge provenance={result.provenance} />

      <div className="flex flex-col items-center gap-2 py-1 text-center">
        <h2 className="sr-only">Model result</h2>
        {/* `Gauge` already renders `severity` as its own colored label below
            the ring (see `components/ui/Gauge.tsx`) -- do not repeat it here. */}
        <Gauge value={result.riskScore} severity={result.severity} />
        <p className="text-lg font-semibold text-ink-primary">{result.diagnosis}</p>
        <p className="max-w-[26rem] text-xs leading-relaxed text-ink-muted">{result.reason}</p>
      </div>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-2 text-xs">
        <dt className="text-ink-faint">Risk score</dt>
        <dd className="text-right font-mono text-ink-primary">{formatPercent(result.riskScore)}</dd>
        <dt className="text-ink-faint">Inference latency</dt>
        <dd className="text-right font-mono text-ink-primary">{formatLatency(result.latencyMs)}</dd>
        <dt className="text-ink-faint">Model status</dt>
        <dd className="text-right font-mono text-ink-primary">{result.provenance.modelSourceLabel ?? "—"}</dd>
      </dl>
    </div>
  );
}
