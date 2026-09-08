"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

/** Progressive disclosure for raw/debug fields -- `backend` (the free-form
 * server tag) is diagnostic information, never a UI decision input. */
export function TechnicalDetails({ result }: { result: DiagnosticResult }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="border-t border-surface-3 pt-3">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between text-xs font-semibold uppercase tracking-wide text-ink-faint"
      >
        Technical details
        <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} aria-hidden="true" />
      </button>
      {open && (
        <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 font-mono text-[11px] text-ink-muted">
          <dt className="text-ink-faint">Backend tag</dt>
          <dd className="truncate text-right text-ink-primary">{result.backend}</dd>
          <dt className="text-ink-faint">Model source</dt>
          <dd className="text-right text-ink-primary">{result.provenance.modelSourceLabel ?? "—"}</dd>
          <dt className="text-ink-faint">Quantum backend</dt>
          <dd className="text-right text-ink-primary">{result.quantumTelemetry.device ?? "—"}</dd>
          <dt className="text-ink-faint">Qubit count</dt>
          <dd className="text-right text-ink-primary">{result.quantumTelemetry.nQubits ?? "—"}</dd>
        </dl>
      )}
    </div>
  );
}
