import { Atom, AlertCircle, ChevronDown } from "lucide-react";
import { PipelineStepBadge } from "@/components/ui/PipelineStepBadge";
import { hasQuantumTelemetry } from "@/lib/quantum-telemetry";
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

const AXIS_TICKS = [-1, -0.5, 0, 0.5, 1] as const;

/**
 * Instrument output, not a trading-app ticker and not a game HUD. Every
 * number here is either read straight from the API response or a preset's
 * own stored value — nothing is interpolated, randomized, or fabricated
 * (see `lib/quantum-telemetry.ts`, unchanged data-layer logic).
 *
 * Whether telemetry DATA exists (`hasQuantumTelemetry`) and whether that
 * data's execution is VERIFIED (`provenance.quantumExecution ===
 * "quantum_simulator"`) are two separate questions, deliberately not
 * collapsed into one boolean:
 *   - A live result with an empty `expectations` array has no data at all
 *     -- the whole panel renders "Quantum telemetry unavailable".
 *   - A live result with real `quantum_execution: "quantum_simulator"` has
 *     data AND a backend attestation that a circuit actually ran -- the
 *     confident accent-colored "QUANTUM SIMULATOR" badge is earned.
 *   - A precomputed demo result has real, stored per-qubit data but NO
 *     attestation this frontend can verify -- the bars still render (the
 *     numbers are real), but the badge reads "NOT INDEPENDENTLY VERIFIABLE"
 *     in neutral, not accent, styling. This is the fix for the exact
 *     contradiction a hostile review caught: the badge must never claim
 *     more than "Quantum Details" below it can back up.
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
  const hasData = hasQuantumTelemetry(telemetry);
  const verified = provenance.quantumExecution === "quantum_simulator";
  const notVerifiable = provenance.quantumExecution === "not_verifiable";
  const qubitCount = telemetry.nQubits ?? telemetry.expectations.length;

  return (
    <div>
      <div className="mb-3 flex items-center justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-faint">
          <span className="relative flex h-2 w-2 shrink-0" aria-hidden="true">
            <span className={cn("h-2 w-2 rounded-full", verified ? "bg-accent" : "bg-ink-faint")} />
          </span>
          <Atom className={verified ? "h-3.5 w-3.5 text-accent" : "h-3.5 w-3.5 text-ink-faint"} aria-hidden="true" />
          Quantum Circuit
        </h3>
        {hasData && (
          <span className="rounded-sm border border-surface-3 bg-surface-0 px-1.5 py-0.5 font-mono text-2xs tabular-nums text-ink-primary">
            {String(qubitCount).padStart(2, "0")} {qubitCount === 1 ? "QUBIT" : "QUBITS"}
          </span>
        )}
      </div>

      {!hasData ? (
        <div className="flex items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4 text-xs text-ink-muted">
          <AlertCircle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          Quantum telemetry unavailable
        </div>
      ) : (
        <>
          <div className="rounded-md border border-surface-3 bg-surface-0 px-3 py-3">
            <div className="mb-2.5 flex items-center justify-between">
              <p className="text-2xs text-ink-faint">Pauli-Z expectation ⟨Z⟩, per qubit</p>
              <AxisLegend />
            </div>

            <div className="flex flex-col gap-2.5">
              {telemetry.expectations.map((value, i) => (
                <QubitRow key={i} index={i} value={value} />
              ))}
            </div>
          </div>

          <div className="mt-2 flex flex-col gap-1">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-ink-faint">
              {telemetry.device && (
                <span>
                  Device: <span className="text-ink-muted">{telemetry.device}</span>
                </span>
              )}
              <span
                className={cn(
                  "inline-flex items-center gap-1 rounded-sm border px-1.5 py-0.5",
                  verified ? "border-accent/30 bg-accent/10 text-accent" : "border-surface-4 bg-surface-2 text-ink-muted",
                )}
              >
                {provenance.quantumExecutionLabel}
              </span>
            </div>
            {notVerifiable && (
              <p className="text-[10px] leading-relaxed text-ink-faint">
                Quantum execution: not independently verifiable for this demo result — these per-qubit values are
                stored with the demo case, not recomputed from a live circuit run.
              </p>
            )}
          </div>
        </>
      )}

      <CircuitArchitecture />

      <QuantumDetails telemetry={telemetry} latencyMs={latencyMs} hasData={hasData} notVerifiable={notVerifiable} />
    </div>
  );
}

/** Tick labels for the shared −1..+1 axis every qubit row is plotted against. */
function AxisLegend() {
  return (
    <div className="hidden font-mono text-[9px] text-ink-faint sm:flex sm:gap-3">
      <span>−1</span>
      <span>0</span>
      <span>+1</span>
    </div>
  );
}

/**
 * One qubit's expectation as a zero-centered, signed horizontal bar on a
 * fixed −1..+1 axis (tick marks at −1, −0.5, 0, 0.5, 1), with a diamond
 * marker at the fill's leading edge as a shape-based state indicator —
 * never color alone. The numeric value and a plain-text magnitude tier
 * repeat the same fact the bar shows, so the row is legible without
 * relying on hue. The fill/marker position transitions when the
 * underlying value changes (a real state transition — switching case or
 * receiving a new result), never on a timer or hover.
 */
function QubitRow({ index, value }: { index: number; value: number }) {
  const clamped = Math.max(-1, Math.min(1, value));
  const magnitude = Math.abs(clamped);
  const tier = magnitude >= 0.6 ? "strong" : magnitude >= 0.25 ? "moderate" : "weak";
  const sign = clamped > 0 ? "+" : clamped < 0 ? "−" : "";
  const markerLeftPct = 50 + clamped * 50;

  return (
    <div className="flex items-center gap-2.5">
      <span className="w-6 shrink-0 font-mono text-xs text-ink-muted">
        Q<sub>{index}</sub>
      </span>

      <div className="relative h-5 flex-1" role="img" aria-label={`Qubit ${index} Pauli-Z expectation ${sign}${magnitude.toFixed(3)}`}>
        {/* signed axis: tick marks at -1, -0.5, 0, 0.5, 1 */}
        <div className="absolute inset-x-0 top-0 flex justify-between" aria-hidden="true">
          {AXIS_TICKS.map((t) => (
            <span key={t} className={cn("w-px", t === 0 ? "h-3 bg-surface-4" : "h-1.5 bg-surface-3")} />
          ))}
        </div>

        <div className="absolute inset-x-0 bottom-0 h-1.5 rounded-sm bg-surface-0" aria-hidden="true">
          <div className="absolute inset-y-0 left-1/2 w-px bg-surface-4" />
          <div
            className={cn(
              "absolute inset-y-0 rounded-sm transition-[width] duration-base",
              clamped >= 0 ? "left-1/2 bg-accent" : "right-1/2 bg-ink-muted",
            )}
            style={{ width: `${magnitude * 50}%` }}
          />
          {/* state marker: a small diamond at the fill's leading edge, distinct from color alone */}
          <div
            className={cn(
              "absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rotate-45 border transition-[left] duration-base",
              clamped >= 0 ? "border-accent bg-accent" : "border-ink-muted bg-ink-muted",
            )}
            style={{ left: `${markerLeftPct}%` }}
            aria-hidden="true"
          />
        </div>
      </div>

      <span className="w-14 shrink-0 text-right font-mono text-xs font-semibold tabular-nums text-ink-primary">
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
      <h4 className="mb-3 text-2xs font-semibold uppercase tracking-wide text-ink-faint">Circuit Architecture</h4>
      <ol className="flex flex-col rounded-md border border-surface-3 bg-surface-0 px-3 py-2.5">
        {ARCHITECTURE_STAGES.map((stage, i) => (
          <li key={stage.name} className="flex gap-3">
            <div className="flex flex-col items-center">
              <PipelineStepBadge index={String(i + 1).padStart(2, "0")} size="sm" />
              {i < ARCHITECTURE_STAGES.length - 1 && (
                <span
                  className="w-px flex-1 border-l border-dashed border-surface-4"
                  aria-hidden="true"
                />
              )}
            </div>
            <div className="pb-3">
              <div className="text-xs font-medium text-ink-primary">{stage.name}</div>
              <div className="text-2xs text-ink-faint">{stage.detail}</div>
            </div>
          </li>
        ))}
      </ol>
      <p className="mt-2 text-2xs text-ink-faint">
        This deployment&apos;s fixed model architecture — not a per-request gate trace. The API does not report
        individual gate sequences or rotation angles for a given prediction.
      </p>
    </div>
  );
}

/**
 * Native `<details>`/`<summary>` — a real disclosure widget, keyboard
 * operable and screen-reader exposed with no custom JS. Every row is
 * either a real value from this result or an explicit "not known" reason —
 * there is no "—" standing in ambiguously for "we didn't bother asking" vs.
 * "the field doesn't exist for this response". The "not known" reason
 * itself is provenance-aware: a live result that simply didn't report a
 * field says "Not reported by API" (an API call happened; this is what it
 * omitted); a precomputed demo result says "Not applicable" (no API call
 * happened at all for this result, so "not reported BY API" would imply a
 * call this frontend never made) — the wording this panel's own badge is
 * checked against by `QuantumTelemetry`'s hostile-review fix.
 */
function QuantumDetails({
  telemetry,
  latencyMs,
  hasData,
  notVerifiable,
}: {
  telemetry: QuantumTelemetryData;
  latencyMs?: number | null;
  hasData: boolean;
  notVerifiable: boolean;
}) {
  const isExactStatevector = telemetry.device === "default.qubit";
  const notReported = notVerifiable ? "Not applicable — no live circuit executed for this demo result" : "Not reported by API";

  const rows: Array<{ label: string; value: string }> = [
    { label: "Number of qubits", value: telemetry.nQubits != null ? String(telemetry.nQubits) : notReported },
    { label: "Circuit depth", value: notReported },
    { label: "Device / backend", value: telemetry.device ?? notReported },
    {
      label: "Shots / statevector",
      value: !telemetry.device
        ? notReported
        : isExactStatevector
          ? "Statevector (exact) — no measurement shots"
          : notReported,
    },
    {
      label: "Execution time",
      value:
        latencyMs != null
          ? `${formatLatency(latencyMs)} (full request, not quantum-only)`
          : notVerifiable
            ? "Not applicable — no live request for this demo result"
            : "Not reported for this result",
    },
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
      {!hasData && (
        <p className="mt-2 text-2xs text-ink-faint">No quantum circuit executed for this result — every field above reflects that.</p>
      )}
      {notVerifiable && (
        <p className="mt-2 text-2xs text-ink-faint">
          The per-qubit values above are real, stored values from this demo case — not fabricated — but nothing in
          this response lets the frontend independently verify a live circuit produced them.
        </p>
      )}
    </details>
  );
}
