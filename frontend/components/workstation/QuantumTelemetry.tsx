import { Atom, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { QuantumTelemetry as QuantumTelemetryData } from "@/lib/types";

const PROVENANCE_LABEL: Record<QuantumTelemetryData["provenance"], string> = {
  live: "LIVE QUANTUM TELEMETRY",
  "precomputed-demo": "PRECOMPUTED DEMO TELEMETRY",
  unavailable: "",
};

export function QuantumTelemetry({ telemetry }: { telemetry: QuantumTelemetryData }) {
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-400">
          <Atom className={cn("h-3.5 w-3.5", telemetry.provenance === "unavailable" ? "text-slate-600" : "text-cyan-400")} />
          Quantum Circuit Telemetry — Pauli-Z Expectations
        </div>
        {telemetry.provenance !== "unavailable" && (
          <span
            className={cn(
              "rounded px-1.5 py-0.5 text-[9px] font-semibold tracking-wide",
              telemetry.provenance === "live" ? "bg-cyan-500/10 text-cyan-400" : "bg-amber-500/10 text-amber-400",
            )}
          >
            {PROVENANCE_LABEL[telemetry.provenance]}
          </span>
        )}
      </div>

      {telemetry.provenance === "unavailable" ? (
        <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-950 px-3 py-4 text-xs text-slate-500">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" />
          Quantum telemetry unavailable
        </div>
      ) : (
        <>
          <div className="grid grid-cols-4 gap-2">
            {telemetry.expectations.map((value, i) => (
              <div key={i} className="rounded-lg border border-slate-800 bg-slate-950 px-2 py-3 text-center">
                <div className="text-[10px] text-slate-500">
                  q<sub>{i}</sub>
                </div>
                <div
                  className={cn(
                    "mt-1 font-mono text-sm font-semibold",
                    value >= 0 ? "text-cyan-400" : "text-rose-400",
                  )}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(3)}
                </div>
                <div className="mt-2 h-1 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className={cn("h-full", value >= 0 ? "bg-cyan-400" : "bg-rose-400")}
                    style={{ width: `${Math.min(100, Math.abs(value) * 100)}%`, marginLeft: value < 0 ? "auto" : 0 }}
                  />
                </div>
              </div>
            ))}
          </div>
          {telemetry.device && (
            <div className="mt-2 text-[10px] text-slate-600">
              {telemetry.nQubits}-qubit circuit · {telemetry.device}
            </div>
          )}
        </>
      )}
    </div>
  );
}
