import { AlertTriangle, FlaskConical, Radio, Database, Share2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProvenanceInfo } from "@/lib/types";

/**
 * The prominent, persistent provenance badge AUDIT.md P1 #5 (D2) asks for —
 * every `DiagnosticResult` carries a `ProvenanceInfo` (see `lib/provenance.ts`),
 * and this is the one place that renders it, so LIVE and MOCK/FALLBACK can
 * never accidentally share the same visual weight the old "Backend: Live/
 * Preset" metric tile gave them (AUDIT.md D2's exact complaint).
 */

const STYLES: Record<
  ProvenanceInfo["provenance"],
  { icon: typeof Radio; className: string }
> = {
  live: {
    icon: Radio,
    className: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  },
  precomputed_demo: {
    icon: FlaskConical,
    className: "border-amber-500/40 bg-amber-500/15 text-amber-300",
  },
  // Deliberately the loudest, highest-contrast styling of the five states —
  // AUDIT.md P1 #7 requirement 5: "Make MOCK/FALLBACK visually unmistakable."
  mock_fallback: {
    icon: AlertTriangle,
    className: "animate-pulse border-rose-500 bg-rose-500/25 text-rose-200",
  },
  cached: {
    icon: Database,
    className: "border-sky-500/40 bg-sky-500/15 text-sky-300",
  },
  proxy: {
    icon: Share2,
    className: "border-slate-500/40 bg-slate-500/15 text-slate-300",
  },
};

export function ProvenanceBadge({ provenance }: { provenance: ProvenanceInfo }) {
  const { icon: Icon, className } = STYLES[provenance.provenance];

  return (
    <div className="flex flex-col gap-1">
      <div
        className={cn(
          "inline-flex w-fit items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-bold tracking-wide",
          className,
        )}
      >
        <Icon className="h-3.5 w-3.5" />
        {provenance.provenanceLabel}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-slate-500">
        {provenance.modelSourceLabel && <span>{provenance.modelSourceLabel}</span>}
        <span>{provenance.quantumExecutionLabel}</span>
      </div>
    </div>
  );
}
