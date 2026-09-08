import { Atom, AlertCircle } from "lucide-react";
import type { ProvenanceInfo, QuantumTelemetry as QuantumTelemetryData } from "@/lib/types";

/**
 * Instrument output, not a trading-app ticker. Availability/trust comes
 * from the single shared `ProvenanceInfo` (`provenance.quantumExecution`)
 * -- this component no longer computes or renders its own provenance
 * label; it only ever displays the actual numbers the API returned (or
 * says plainly that none exist). Never hardcodes a value, never derives
 * one from `risk_score`, never substitutes a preset's numbers into a live
 * result (see `lib/quantum-telemetry.ts`, unchanged data-layer logic).
 */
export function QuantumTelemetry({
  telemetry,
  provenance,
}: {
  telemetry: QuantumTelemetryData;
  provenance: ProvenanceInfo;
}) {
  const available = provenance.quantumExecution === "quantum_simulator" && telemetry.expectations.length > 0;

  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-ink-muted">
        <Atom className={available ? "h-3.5 w-3.5 text-accent" : "h-3.5 w-3.5 text-ink-faint"} aria-hidden="true" />
        Quantum circuit telemetry — Pauli-Z expectations
      </div>

      {!available ? (
        <div className="flex items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4 text-xs text-ink-muted">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          Quantum telemetry unavailable
        </div>
      ) : (
        <>
          <div className="grid grid-cols-4 gap-2">
            {telemetry.expectations.map((value, i) => (
              <div key={i} className="rounded-md border border-surface-3 bg-surface-0 px-2 py-3 text-center">
                <div className="text-[10px] text-ink-faint">
                  q<sub>{i}</sub>
                </div>
                <div className="mt-1 font-mono text-sm font-semibold text-ink-primary">
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(3)}
                </div>
                <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-surface-3">
                  <div
                    className="h-full bg-accent"
                    style={{ width: `${Math.min(100, Math.abs(value) * 100)}%`, marginLeft: value < 0 ? "auto" : 0 }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-3 text-[10px] text-ink-faint">
            <span>{telemetry.nQubits}-qubit circuit</span>
            {telemetry.device && <span>Device: {telemetry.device}</span>}
            <span>Backend: {provenance.quantumExecutionLabel}</span>
          </div>
        </>
      )}
    </div>
  );
}
