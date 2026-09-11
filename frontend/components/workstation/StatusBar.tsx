import { PLANE_LABELS } from "@/lib/viewer";
import type { DiagnosticResult } from "@/lib/types";

/**
 * BOTTOM zone: a persistent technical-status strip, replacing the old
 * `TechnicalDetails.tsx` (a collapsed-by-default accordion buried inside
 * the analysis column). Just always visible, where a real PACS/analysis
 * workstation's status bar lives, instead of hidden behind a disclosure
 * toggle competing for space with the AI panel above it.
 *
 * Deliberately does NOT render `result.backend` -- that field is a raw
 * internal tag (e.g. "mock/preset", "cache-fallback/case_0003") meant for
 * debugging, not a second, competing provenance vocabulary alongside the
 * canonical LIVE/PRECOMPUTED DEMO/MOCK-FALLBACK labels `ProvenanceBadge`
 * already renders. A hostile review caught this leaking to end users as
 * "Backend tag: mock/preset" -- developer jargon sitting right next to a
 * loud, correctly-worded "PRECOMPUTED DEMO" badge, risking a viewer
 * reading "mock" as "this whole result is fake" even though the result is
 * exactly what it claims to be.
 */
export function StatusBar({ result }: { result: DiagnosticResult | null }) {
  if (!result) {
    return (
      <div className="no-print shrink-0 border-t border-surface-3 bg-surface-1 px-4 py-1.5 text-2xs text-ink-faint sm:px-6">
        No study loaded — acquisition metadata appears here once a result is available.
      </div>
    );
  }

  const items: Array<{ label: string; value: string }> = [
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
      className="no-print flex shrink-0 flex-wrap items-center gap-x-6 gap-y-1 border-t border-surface-3 bg-surface-1 px-4 py-1.5 font-mono text-2xs text-ink-muted sm:px-6"
    >
      {items.map((item) => (
        <span key={item.label}>
          <span className="text-ink-faint">{item.label}:</span> <span className="text-ink-primary">{item.value}</span>
        </span>
      ))}
    </div>
  );
}
