import { AlertTriangle, FlaskConical, Radio, Database, Share2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProvenanceInfo } from "@/lib/types";

/**
 * The ONLY component in the product allowed to render a provenance label.
 * `QuantumTelemetry`, `ReportExport`, and every result surface consume
 * this component (or read `ProvenanceInfo` fields directly for prose, e.g.
 * "Quantum backend: {provenance.quantumExecutionLabel}") -- none of them
 * format their own copy of the label. `provenance` (LIVE / PRECOMPUTED
 * DEMO / MOCK-FALLBACK / CACHED / PROXY) and `quantumExecution` (QUANTUM
 * SIMULATOR / UNAVAILABLE) are two independent facts, not opposites: a
 * genuinely LIVE result normally pairs with "QUANTUM SIMULATOR" -- that
 * pairing means a real VQC executed on a simulator backend, which is this
 * project's correct, expected, non-degraded state, never mislabeled
 * "fake" for using a simulator.
 */

const STYLES: Record<ProvenanceInfo["provenance"], { icon: typeof Radio; className: string }> = {
  live: { icon: Radio, className: "border-status-live/40 bg-status-live/10 text-status-live" },
  precomputed_demo: { icon: FlaskConical, className: "border-status-demo/40 bg-status-demo/10 text-status-demo" },
  // The loudest, highest-contrast state on purpose (AUDIT.md P1 #7
  // requirement 5) -- distinguished by icon + text + a persistent alert
  // ring, not by color alone and not by animation alone.
  mock_fallback: {
    icon: AlertTriangle,
    className: "border-status-fallback bg-status-fallback/15 text-status-fallback shadow-alert",
  },
  cached: { icon: Database, className: "border-status-demo/40 bg-status-demo/10 text-status-demo" },
  proxy: { icon: Share2, className: "border-status-proxy/40 bg-status-proxy/10 text-status-proxy" },
};

export function ProvenanceBadge({ provenance, compact = false }: { provenance: ProvenanceInfo; compact?: boolean }) {
  const { icon: Icon, className } = STYLES[provenance.provenance];

  const badge = (
    <span
      className={cn(
        "inline-flex w-fit items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-bold tracking-wide",
        className,
      )}
      role="status"
    >
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {provenance.provenanceLabel}
    </span>
  );

  if (compact) return badge;

  return (
    <div className="flex flex-col gap-1">
      {badge}
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-ink-faint">
        {provenance.modelSourceLabel && <span>Model: {provenance.modelSourceLabel}</span>}
        <span>Quantum backend: {provenance.quantumExecutionLabel}</span>
      </div>
    </div>
  );
}
