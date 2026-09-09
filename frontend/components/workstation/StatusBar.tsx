import { PLANE_LABELS } from "@/lib/viewer";
import type { DiagnosticResult } from "@/lib/types";

/**
 * BOTTOM zone: a persistent technical-status strip, replacing the old
 * `TechnicalDetails.tsx` (a collapsed-by-default accordion buried inside
 * the analysis column). Same underlying fields — `backend` is diagnostic
 * information only, never a UI decision input — just always visible,
 * where a real PACS/analysis workstation's status bar lives, instead of
 * hidden behind a disclosure toggle competing for space with the AI
 * panel above it.
 */
export function StatusBar({ result }: { result: DiagnosticResult | null }) {
  if (!result) {
    return (
      <div className="border-t border-surface-3 bg-surface-1 px-4 py-1.5 text-2xs text-ink-faint sm:px-6">
        No study loaded — acquisition metadata appears here once a result is available.
      </div>
    );
  }

  const items: Array<{ label: string; value: string }> = [
    { label: "Backend tag", value: result.backend },
    { label: "Model", value: result.provenance.modelSourceLabel ?? "—" },
    { label: "Quantum backend", value: result.quantumTelemetry.device ?? "—" },
    { label: "Qubits", value: result.quantumTelemetry.nQubits != null ? String(result.quantumTelemetry.nQubits) : "—" },
    { label: "Plane", value: PLANE_LABELS[result.volume.primaryPlane] },
    { label: "Primary slice", value: String(result.volume.primarySliceIndex + 1) },
  ];

  return (
    <div
      role="contentinfo"
      aria-label="Technical status and acquisition metadata"
      className="flex flex-wrap items-center gap-x-6 gap-y-1 border-t border-surface-3 bg-surface-1 px-4 py-1.5 font-mono text-2xs text-ink-muted sm:px-6"
    >
      {items.map((item) => (
        <span key={item.label}>
          <span className="text-ink-faint">{item.label}:</span> <span className="text-ink-primary">{item.value}</span>
        </span>
      ))}
    </div>
  );
}
