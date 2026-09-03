import { Atom } from "lucide-react";
import { cn } from "@/lib/utils";

export function QuantumTelemetry({ expectations }: { expectations: [number, number, number, number] }) {
  return (
    <div>
      <div className="mb-3 flex items-center gap-1.5 text-xs font-medium text-slate-400">
        <Atom className="h-3.5 w-3.5 text-cyan-400" />
        Quantum Circuit Telemetry — Pauli-Z Expectations
      </div>
      <div className="grid grid-cols-4 gap-2">
        {expectations.map((value, i) => (
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
    </div>
  );
}
