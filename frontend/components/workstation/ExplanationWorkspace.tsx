"use client";

import { useEffect, useRef, useState } from "react";
import {
  X,
  Eye,
  Layers,
  Info,
  AlertTriangle,
  Boxes,
  Minimize2,
  Atom,
  ListChecks,
  ScanLine,
} from "lucide-react";
import { Switch } from "@/components/ui/Switch";
import { Surface } from "@/components/ui/Surface";
import { ProvenanceBadge } from "@/components/workstation/ProvenanceBadge";
import { QuantumTelemetry } from "@/components/workstation/QuantumTelemetry";
import { PLANE_LABELS, sliceImageSrc, toImageSrc } from "@/lib/viewer";
import { cn, formatLatency, formatPercent } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

const SEVERITY_CLASS: Record<DiagnosticResult["severity"], string> = {
  Normal: "text-severity-normal",
  Indeterminate: "text-severity-indeterminate",
  "Urgent Surgical Consult": "text-severity-urgent",
};

const SECTIONS = [
  { id: "visual-evidence", index: 1, label: "Visual evidence", icon: ScanLine },
  { id: "model-output", index: 2, label: "Model output", icon: Layers },
  { id: "quantum-contribution", index: 3, label: "Quantum contribution", icon: Atom },
  { id: "technical-context", index: 4, label: "Technical context", icon: Boxes },
  { id: "limitations", index: 5, label: "Limitations", icon: AlertTriangle },
] as const;

/**
 * The full explainability surface, opened from `ExplanationPanel`'s "Open
 * Explanation Workspace" action. Five numbered sections answer one
 * question each -- what did we see, what did the model output, what did
 * the quantum circuit contribute, how is the pipeline built, and what are
 * this result's real limits -- so a technical reviewer can audit the
 * chain of evidence instead of being handed a single unexplained heatmap.
 *
 * Every number and claim here is read from the same `DiagnosticResult` /
 * `ProvenanceInfo` the rest of the workstation renders -- no separate
 * "explanation mode" data path exists to drift out of sync with the
 * viewer, the prediction panel, or the quantum telemetry panel. Section 1
 * reuses `lib/viewer.ts` exactly as `ExplanationPanel`/`MRIViewer` do
 * (never a re-generated or duplicate image asset). Section 3 renders the
 * actual `QuantumTelemetry` component rather than a second copy of its
 * qubit-bar logic, so "what the quantum circuit contributed" can never
 * show different numbers here than in the sidebar.
 */
export function ExplanationWorkspace({
  open,
  onClose,
  result,
}: {
  open: boolean;
  onClose: () => void;
  result: DiagnosticResult;
}) {
  const [overlayOn, setOverlayOn] = useState(true);
  const [opacity, setOpacity] = useState(70);
  const sectionRefs = useRef<Record<string, HTMLElement | null>>({});
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  const volume = result.volume;
  const hasOverlay = Boolean(volume.gradcamOverlay && volume.gradcamPlane !== null);
  const gradcamPlane = volume.gradcamPlane;
  const gradcamSliceIndex = volume.gradcamSliceIndex ?? 0;
  const baseAtGradcamSlice = hasOverlay && gradcamPlane ? sliceImageSrc(volume, gradcamPlane, gradcamSliceIndex) : null;
  const numSlicesAtPlane = gradcamPlane ? volume.planes[gradcamPlane].numSlices : 0;

  function jumpTo(id: string) {
    sectionRefs.current[id]?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-surface-0/85 p-3 animate-fade-in sm:p-6"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="explanation-workspace-title"
        className="flex h-full w-full max-w-5xl flex-col overflow-hidden rounded-lg border border-surface-3 bg-surface-1 shadow-3"
      >
        <div className="flex items-start justify-between gap-4 border-b border-surface-3 px-5 py-4 sm:px-6">
          <div>
            <p className="font-mono text-2xs font-semibold uppercase tracking-[0.2em] text-accent">
              Explanation Workspace
            </p>
            <p className="mt-1.5 text-2xs font-semibold uppercase tracking-wide text-ink-faint">Prediction</p>
            <h2 id="explanation-workspace-title" className="mt-0.5 text-lg font-bold text-ink-primary">
              {result.diagnosis} <span className="text-ink-faint">—</span>{" "}
              <span className={SEVERITY_CLASS[result.severity]}>{result.severity}</span>
            </h2>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            <div className="hidden sm:block">
              <ProvenanceBadge provenance={result.provenance} compact />
            </div>
            <button
              onClick={onClose}
              aria-label="Close explanation workspace"
              className="text-ink-muted hover:text-ink-primary"
            >
              <X className="h-5 w-5" aria-hidden="true" />
            </button>
          </div>
        </div>

        <nav
          aria-label="Explanation sections"
          className="flex gap-1 overflow-x-auto border-b border-surface-3 bg-surface-0 px-3 py-2 sm:px-5"
        >
          {SECTIONS.map((section) => (
            <button
              key={section.id}
              type="button"
              onClick={() => jumpTo(section.id)}
              className="flex shrink-0 items-center gap-1.5 rounded-sm px-2.5 py-1.5 text-2xs font-medium text-ink-muted transition-colors hover:bg-surface-2 hover:text-ink-primary"
            >
              <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-[3px] border border-surface-4 font-mono text-[9px] text-ink-faint">
                {section.index}
              </span>
              {section.label}
            </button>
          ))}
        </nav>

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <p className="border-b border-surface-3 px-5 py-3 font-mono text-2xs font-semibold uppercase tracking-[0.2em] text-ink-faint sm:px-6">
            Why?
          </p>

          <Section id="visual-evidence" refs={sectionRefs} index={1} title="Visual evidence" icon={ScanLine}>
            {hasOverlay && gradcamPlane ? (
              <div className="grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1fr)_272px]">
                <div className="relative aspect-square w-full overflow-hidden rounded-md border border-surface-3 bg-black">
                  {baseAtGradcamSlice && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={baseAtGradcamSlice} alt="" aria-hidden="true" className="absolute inset-0 h-full w-full object-contain" />
                  )}
                  {overlayOn && volume.gradcamOverlay && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={toImageSrc(volume.gradcamOverlay)}
                      alt="Grad-CAM model attention visualization"
                      className="absolute inset-0 h-full w-full object-contain transition-opacity duration-fast"
                      style={{ opacity: opacity / 100 }}
                    />
                  )}
                  <div className="pointer-events-none absolute left-3 top-3 rounded bg-surface-0/85 px-2 py-1 font-mono text-[11px] leading-tight text-ink-primary">
                    <div>
                      Explained slice {numSlicesAtPlane > 0 ? gradcamSliceIndex + 1 : 0} / {numSlicesAtPlane}
                    </div>
                    <div className="text-ink-faint">Target: {result.diagnosis}</div>
                  </div>
                </div>

                <div className="flex flex-col gap-4">
                  <div>
                    <div className="mb-1.5 flex items-center justify-between gap-2">
                      <span className="flex items-center gap-1.5 text-xs text-ink-muted">
                        <Layers className="h-3.5 w-3.5" aria-hidden="true" />
                        Grad-CAM overlay
                      </span>
                      <Switch checked={overlayOn} onChange={() => setOverlayOn((v) => !v)} label="Toggle Grad-CAM overlay" />
                    </div>
                    <label htmlFor="workspace-opacity" className="mb-1 flex items-center justify-between text-2xs text-ink-faint">
                      <span>Overlay opacity</span>
                      <span className="font-mono text-ink-primary">{opacity}%</span>
                    </label>
                    <input
                      id="workspace-opacity"
                      type="range"
                      min={0}
                      max={100}
                      value={opacity}
                      disabled={!overlayOn}
                      onChange={(e) => setOpacity(Number(e.target.value))}
                      aria-valuetext={`${opacity} percent`}
                      className="w-full accent-accent disabled:opacity-40"
                    />
                  </div>

                  <div>
                    <p className="mb-1.5 text-2xs font-semibold uppercase tracking-wide text-ink-faint">Legend</p>
                    <div
                      className="h-2 w-full rounded-full"
                      style={{ background: "linear-gradient(to right, #38bdf8, #facc15, #ef4444)" }}
                      aria-hidden="true"
                    />
                    <div className="mt-1 flex justify-between text-2xs text-ink-faint">
                      <span>Low attention</span>
                      <span>High attention</span>
                    </div>
                    <p className="mt-1 text-2xs text-ink-faint">
                      Relative within this slice only — not calibrated to a fixed intensity scale across cases.
                    </p>
                  </div>

                  <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-2xs">
                    <dt className="text-ink-faint">Target</dt>
                    <dd className="text-right text-ink-primary">{result.diagnosis}</dd>
                    <dt className="text-ink-faint">Explained slice</dt>
                    <dd className="text-right font-mono text-ink-primary">
                      {numSlicesAtPlane > 0 ? gradcamSliceIndex + 1 : 0} / {numSlicesAtPlane}
                    </dd>
                    <dt className="text-ink-faint">Plane</dt>
                    <dd className="text-right text-ink-primary">{PLANE_LABELS[gradcamPlane]}</dd>
                  </dl>

                  <div className="rounded-md border border-surface-3 bg-surface-0 px-3 py-2.5">
                    <p className="flex items-start gap-1.5 text-2xs leading-relaxed text-ink-muted">
                      <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" aria-hidden="true" />
                      <span>
                        This is a <strong className="text-ink-primary">model attention visualization</strong> — it
                        shows which image regions most influenced the ResNet18 embedding, computed for one
                        representative slice only. It is not a segmentation and does not prove a lesion exists;
                        independent radiologist review is required.
                      </span>
                    </p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-ink-muted">No Grad-CAM overlay was computed for this result.</p>
            )}
          </Section>

          <Section id="model-output" refs={sectionRefs} index={2} title="Model output" icon={Layers}>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
              <Stat label="Risk score" value={formatPercent(result.riskScore)} />
              <Stat label="Severity" value={result.severity} valueClassName={SEVERITY_CLASS[result.severity]} />
              <Stat label="Diagnosis" value={result.diagnosis} />
              <Stat label="Inference latency" value={formatLatency(result.latencyMs)} />
              <Stat label="Model status" value={result.provenance.modelSourceLabel ?? "—"} />
            </div>
            <div className="mt-4">
              <ProvenanceBadge provenance={result.provenance} />
            </div>
            <p className="mt-3 text-2xs text-ink-faint">
              This is the model&apos;s raw output for this case — a risk probability and severity band, not a
              confirmed diagnosis.
            </p>
          </Section>

          <Section id="quantum-contribution" refs={sectionRefs} index={3} title="Quantum contribution" icon={Atom}>
            <div className="rounded-md border border-surface-3 bg-surface-0 px-3 py-2.5 mb-4">
              <p className="flex items-start gap-1.5 text-2xs leading-relaxed text-ink-muted">
                <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" aria-hidden="true" />
                <span>
                  This architecture&apos;s classical readout is a linear layer + sigmoid applied directly to these
                  four Pauli-Z expectation values — so they are, architecturally, the exact numeric inputs the risk
                  score above was computed from. That is the extent of what this design supports claiming; no single
                  qubit or gate is individually credited with &quot;causing&quot; the diagnosis.
                </span>
              </p>
            </div>
            <QuantumTelemetry telemetry={result.quantumTelemetry} provenance={result.provenance} latencyMs={result.latencyMs} />
          </Section>

          <Section id="technical-context" refs={sectionRefs} index={4} title="Technical context" icon={Boxes}>
            <Surface className="overflow-hidden">
              <dl className="divide-y divide-surface-3">
                <TechRow icon={Boxes} label="Feature extractor" value="ResNet18 (ImageNet-pretrained CNN backbone)" />
                <TechRow icon={Layers} label="Embedding dimension" value="512" />
                <TechRow icon={Minimize2} label="Compression" value="PCA → 4 scalars, scaled to rotation angles" />
                <TechRow icon={Eye} label="Quantum input dimension" value="4" />
                <TechRow icon={Atom} label="Quantum circuit" value="4-qubit VQC — 3 variational layers, ring entanglement, PennyLane default.qubit simulator" />
              </dl>
            </Surface>
          </Section>

          <Section id="limitations" refs={sectionRefs} index={5} title="Limitations" icon={AlertTriangle} last>
            <ul className="space-y-2.5 rounded-md border border-warning/30 bg-warning/10 px-4 py-3.5">
              <LimitationItem
                title="Small research dataset"
                detail="Evaluated on real RSNA Knee data, n=58 studies — not large enough to establish generalization."
              />
              <LimitationItem
                title="No clinical validation"
                detail="This has not undergone clinical trials and is not a validated diagnostic tool."
              />
              <LimitationItem
                title="Simulator-based quantum execution"
                detail="The quantum circuit runs on PennyLane's default.qubit classical simulator, not physical quantum hardware."
              />
              <LimitationItem
                title="Research prototype"
                detail="Investigational output only. Findings require independent review by a licensed radiologist or orthopedic clinician."
              />
            </ul>
          </Section>
        </div>
      </div>
    </div>
  );
}

function Section({
  id,
  refs,
  index,
  title,
  icon: Icon,
  children,
  last,
}: {
  id: string;
  refs: React.MutableRefObject<Record<string, HTMLElement | null>>;
  index: number;
  title: string;
  icon: typeof ScanLine;
  children: React.ReactNode;
  last?: boolean;
}) {
  return (
    <section
      id={id}
      ref={(el) => {
        refs.current[id] = el;
      }}
      className={cn("scroll-mt-3 px-5 py-6 sm:px-6", !last && "border-b border-surface-3")}
    >
      <div className="mb-4 flex items-center gap-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-sm border border-surface-4 font-mono text-xs text-ink-muted">
          {index}
        </span>
        <Icon className="h-4 w-4 text-accent" aria-hidden="true" />
        <h3 className="text-sm font-semibold uppercase tracking-wide text-ink-primary">{title}</h3>
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value, valueClassName }: { label: string; value: string; valueClassName?: string }) {
  return (
    <div className="rounded-md border border-surface-3 bg-surface-0 px-3 py-2.5">
      <p className="text-2xs text-ink-faint">{label}</p>
      <p className={cn("mt-1 truncate font-mono text-sm font-semibold text-ink-primary", valueClassName)}>{value}</p>
    </div>
  );
}

function TechRow({ icon: Icon, label, value }: { icon: typeof Boxes; label: string; value: string }) {
  return (
    <div className="flex items-start gap-3 px-4 py-2.5 text-xs">
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-ink-faint" aria-hidden="true" />
      <div className="min-w-0 flex-1">
        <dt className="text-2xs text-ink-faint">{label}</dt>
        <dd className="mt-0.5 break-words font-mono text-ink-primary">{value}</dd>
      </div>
    </div>
  );
}

function LimitationItem({ title, detail }: { title: string; detail: string }) {
  return (
    <li className="flex items-start gap-2 text-xs">
      <ListChecks className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warning" aria-hidden="true" />
      <span>
        <span className="font-semibold text-ink-primary">{title}.</span> <span className="text-ink-muted">{detail}</span>
      </span>
    </li>
  );
}
