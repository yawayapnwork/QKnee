import { Atom, AlertCircle, ChevronDown } from "lucide-react";
import type { ProvenanceInfo, QuantumTelemetry as QuantumTelemetryData } from "@/lib/types";
import { formatLatency } from "@/lib/utils";
import { cn } from "@/lib/utils";

const ARCHITECTURE_STAGES = [
  { name: "Input", detail: "Pooled per-slice feature vector" },
  { name: "Angle Encoding", detail: "Features mapped to rotation angles" },
  { name: "Rotation Layer", detail: "Parameterized single-qubit rotations" },
  { name: "Entanglement", detail: "Strongly-entangling multi-qubit gates" },
  { name: "Measurement", detail: "Pauli-Z expectation per qubit" },
] as const;

/**
 * Instrument output, not a trading-app ticker and not a game HUD. Every
 * number here is either read straight from the API response or explicitly
 * labeled "not reported" — nothing is interpolated, randomized, or reused
 * from a preset to stand in for a live result (see `lib/quantum-telemetry.ts`,
 * unchanged data-layer logic). Availability/trust comes from the single
 * shared `ProvenanceInfo` (`provenance.quantumExecution`); this component
 * only ever renders the actual numbers the API returned or says plainly
 * that none exist.
 *
 * The five-stage flow below (Input → Angle Encoding → Rotation Layer →
 * Entanglement → Measurement) is this deployment's fixed VQC architecture
 * (`extras/models/vqc_strongly_entangling.py`: `AngleEmbedding` +
 * `StronglyEntanglingLayers` + per-qubit `PauliZ`) — a real, simplified
 * diagram of how the circuit is built, not a fabricated gate-by-gate trace.
 * The API reports no per-request gate list or layer count, so this stays a
 * static architecture diagram rather than inventing one.
 */
export function QuantumTelemetry({
  telemetry,
  provenance,
  latencyMs,
}: {
  telemetry: QuantumTelemetryData;
  provenance: ProvenanceInfo;
  latencyMs?: number | null;
}) {
  const available = provenance.quantumExecution === "quantum_simulator" && telemetry.expectations.length > 0;
  const qubitCount = telemetry.nQubits ?? telemetry.expectations.length;

  return (
    <div>
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-faint">
          <Atom className={available ? "h-3.5 w-3.5 text-accent" : "h-3.5 w-3.5 text-ink-faint"} aria-hidden="true" />
          Quantum Telemetry
        </div>
        {available && (
          <span className="font-mono text-2xs text-ink-faint">
            {qubitCount} {qubitCount === 1 ? "QUBIT" : "QUBITS"}
          </span>
        )}
      </div>

      {!available ? (
        <div className="flex items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4 text-xs text-ink-muted">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          Quantum telemetry unavailable
        </div>
      ) : (
        <>
          <p className="mb-3 text-2xs text-ink-faint">
            Per-qubit Pauli-Z expectation, read directly from the executed circuit. Range −1 to +1.
          </p>

          <div className="flex flex-col gap-2">
            {telemetry.expectations.map((value, i) => (
              <QubitRow key={i} index={i} value={value} />
            ))}
          </div>

          <div className="mt-2 flex flex-wrap gap-x-3 text-[10px] text-ink-faint">
            {telemetry.device && <span>Device: {telemetry.device}</span>}
            <span>Backend: {provenance.quantumExecutionLabel}</span>
          </div>
        </>
      )}

      <CircuitArchitecture />

      <QuantumDetails telemetry={telemetry} latencyMs={latencyMs} available={available} />
    </div>
  );
}

/**
 * One qubit's expectation as a zero-centered, signed horizontal bar — a
 * fill running left from center for negative values and right for
 * positive, on a fixed −1..+1 axis. The numeric value and a plain-text
 * magnitude tier (never color alone) both encode the same fact the bar
 * shows, so the row is legible without relying on hue. The fill width
 * transitions when the underlying value changes (a real state transition
 * — switching case or receiving a new result), never on a timer or hover.
 */
function QubitRow({ index, value }: { index: number; value: number }) {
  const clamped = Math.max(-1, Math.min(1, value));
  const magnitude = Math.abs(clamped);
  const tier = magnitude >= 0.6 ? "strong" : magnitude >= 0.25 ? "moderate" : "weak";
  const sign = clamped > 0 ? "+" : clamped < 0 ? "−" : "";

  return (
    <div className="flex items-center gap-2.5">
      <span className="w-6 shrink-0 font-mono text-xs text-ink-faint">
        Q<sub>{index}</sub>
      </span>

      <div className="relative h-4 flex-1 rounded-sm bg-surface-0" aria-hidden="true">
        <div className="absolute inset-y-0 left-1/2 w-px bg-surface-4" />
        <div
          className={cn(
            "absolute inset-y-0 rounded-sm transition-[width] duration-base",
            clamped >= 0 ? "left-1/2 bg-accent" : "right-1/2 bg-ink-muted",
          )}
          style={{ width: `${magnitude * 50}%` }}
        />
      </div>

      <span className="w-14 shrink-0 text-right font-mono text-xs font-semibold text-ink-primary">
        {sign}
        {magnitude.toFixed(3).replace("-", "")}
      </span>
      <span className="w-16 shrink-0 text-[10px] uppercase tracking-wide text-ink-faint">{tier}</span>
    </div>
  );
}

function CircuitArchitecture() {
  return (
    <div className="mt-4 border-t border-surface-3 pt-4">
      <h4 className="mb-2 text-2xs font-semibold uppercase tracking-wide text-ink-faint">Circuit Architecture</h4>
      <ol className="flex flex-col">
        {ARCHITECTURE_STAGES.map((stage, i) => (
          <li key={stage.name} className="flex gap-3">
            <div className="flex flex-col items-center">
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-surface-4 font-mono text-[10px] text-ink-muted">
                {i + 1}
              </span>
              {i < ARCHITECTURE_STAGES.length - 1 && <span className="w-px flex-1 bg-surface-3" aria-hidden="true" />}
            </div>
            <div className="pb-3">
              <div className="text-xs font-medium text-ink-primary">{stage.name}</div>
              <div className="text-2xs text-ink-faint">{stage.detail}</div>
            </div>
          </li>
        ))}
      </ol>
      <p className="text-2xs text-ink-faint">
        This deployment&apos;s fixed model architecture — not a per-request gate trace. The API does not report
        individual gate sequences or rotation angles for a given prediction.
      </p>
    </div>
  );
}

/**
 * Native `<details>`/`<summary>` — a real disclosure widget, keyboard
 * operable and screen-reader exposed with no custom JS. Every row is
 * either a real value from this result or an explicit "Not reported by
 * API" — there is no "—" standing in ambiguously for "we didn't bother
 * asking" vs. "the field doesn't exist for this response".
 */
function QuantumDetails({
  telemetry,
  latencyMs,
  available,
}: {
  telemetry: QuantumTelemetryData;
  latencyMs?: number | null;
  available: boolean;
}) {
  const isExactStatevector = telemetry.device === "default.qubit";

  const rows: Array<{ label: string; value: string }> = [
    { label: "Number of qubits", value: telemetry.nQubits != null ? String(telemetry.nQubits) : "Not reported by API" },
    { label: "Circuit depth", value: "Not reported by API" },
    { label: "Device / backend", value: telemetry.device ?? "Not reported by API" },
    {
      label: "Shots / statevector",
      value: !telemetry.device
        ? "Not reported by API"
        : isExactStatevector
          ? "Statevector (exact) — no measurement shots"
          : "Not reported by API",
    },
    { label: "Execution time", value: latencyMs != null ? `${formatLatency(latencyMs)} (full request, not quantum-only)` : "Not reported for this result" },
    { label: "Model version", value: "Not exposed by the API" },
  ];

  return (
    <details className="mt-4 border-t border-surface-3 pt-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 text-2xs font-semibold uppercase tracking-wide text-ink-faint [&::-webkit-details-marker]:hidden">
        Quantum Details
        <ChevronDown className="h-3.5 w-3.5 shrink-0 transition-transform duration-fast [details[open]_&]:rotate-180" aria-hidden="true" />
      </summary>
      <dl className="mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-2xs">
        {rows.map((row) => (
          <div className="contents" key={row.label}>
            <dt className="text-ink-faint">{row.label}</dt>
            <dd className="text-right font-mono text-ink-primary">{row.value}</dd>
          </div>
        ))}
      </dl>
      {!available && (
        <p className="mt-2 text-2xs text-ink-faint">No quantum circuit executed for this result — every field above reflects that.</p>
      )}
    </details>
  );
}
