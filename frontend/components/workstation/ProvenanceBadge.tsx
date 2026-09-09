import { AlertTriangle, FlaskConical, Radio, Database, Share2 } from "lucide-react";
import { StatusIndicator, type StatusTone } from "@/components/ui/StatusIndicator";
import type { ProvenanceInfo } from "@/lib/types";

/**
 * The ONE component in the product allowed to render a provenance label.
 * `QuantumTelemetry`, `ReportExport`, and every result surface consume
 * this component (or read `ProvenanceInfo` fields directly for prose) --
 * none format their own copy of the label. This is an EXHAUSTIVE mapping
 * (`Record<ProvenanceInfo["provenance"], ...>` -- TypeScript errors if a
 * new provenance value is ever added without a matching tone here), which
 * is what makes "MOCK cannot accidentally render as LIVE" a structural
 * guarantee rather than a convention: there is no code path through which
 * `mock_fallback` could resolve to `StatusIndicator`'s `"live"` tone.
 *
 * `provenance` (LIVE / PRECOMPUTED DEMO / MOCK-FALLBACK / CACHED / PROXY)
 * and `quantumExecution` (QUANTUM SIMULATOR / NOT INDEPENDENTLY VERIFIABLE /
 * QUANTUM TELEMETRY UNAVAILABLE) are two independent facts, not opposites --
 * a genuinely LIVE result normally pairs with "QUANTUM SIMULATOR": a real
 * VQC executed on a simulator backend, this project's correct, expected,
 * non-degraded state. A PRECOMPUTED DEMO result pairs with "NOT
 * INDEPENDENTLY VERIFIABLE" -- real stored per-qubit numbers are still
 * shown, but this frontend has no artifact proving a circuit produced them
 * for this specific result, so it must not borrow the confident
 * "QUANTUM SIMULATOR" wording a backend-attested execution earns.
 */
const PROVENANCE_TONE: Record<ProvenanceInfo["provenance"], { tone: StatusTone; icon: typeof Radio }> = {
  live: { tone: "live", icon: Radio },
  precomputed_demo: { tone: "demo", icon: FlaskConical },
  mock_fallback: { tone: "fallback", icon: AlertTriangle },
  cached: { tone: "demo", icon: Database },
  proxy: { tone: "proxy", icon: Share2 },
};

export function ProvenanceBadge({ provenance, compact = false }: { provenance: ProvenanceInfo; compact?: boolean }) {
  const { tone, icon } = PROVENANCE_TONE[provenance.provenance];

  if (compact) {
    return <StatusIndicator tone={tone} label={provenance.provenanceLabel} icon={icon} compact />;
  }

  return (
    <div className="flex flex-col gap-1">
      <StatusIndicator tone={tone} label={provenance.provenanceLabel} icon={icon} />
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-2xs text-ink-faint">
        {provenance.modelSourceLabel && <span>Model: {provenance.modelSourceLabel}</span>}
        <span>Quantum execution: {provenance.quantumExecutionLabel}</span>
      </div>
    </div>
  );
}
