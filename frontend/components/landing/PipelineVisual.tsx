import { Scan, Boxes, Minimize2, Atom, Eye, ChevronRight } from "lucide-react";
import { Surface } from "@/components/ui/Surface";
import { Gauge } from "@/components/ui/Gauge";

/**
 * The hero's proof-of-substance: a real instrument-panel rendering of the
 * six-stage pipeline (matches `qknee/models/pipeline.py` stage numbering),
 * not a particle system or a stock-photo collage. Every number here is
 * illustrative/static -- it is not wired to a live case -- and is labeled
 * as such, the same way `QuantumTelemetry.tsx` never fabricates numbers
 * for a case that hasn't run. The four Pauli-Z bars and the risk gauge
 * exist to show *what kind of output* each stage produces, not a real
 * prediction.
 */
const ILLUSTRATIVE_EXPECTATIONS = [0.62, -0.18, 0.41, -0.05];

function StageArrow() {
  return (
    <div className="flex items-center justify-center px-1 text-ink-faint sm:px-0" aria-hidden="true">
      <ChevronRight className="h-4 w-4 rotate-90 sm:rotate-0" />
    </div>
  );
}

export function PipelineVisual() {
  return (
    <Surface className="overflow-hidden">
      <div className="border-b border-surface-3 px-4 py-2.5 text-2xs font-semibold uppercase tracking-wide text-ink-faint">
        Pipeline — illustrative, one forward pass
      </div>
      <div className="overflow-x-auto">
      <div className="grid grid-cols-1 items-stretch gap-0 p-4 sm:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr_auto_1fr] sm:min-w-[920px] sm:p-6">
        {/* 1. MRI volume */}
        <div className="flex flex-col items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4">
          <Scan className="h-4 w-4 text-accent" aria-hidden="true" />
          <svg viewBox="0 0 64 64" className="h-14 w-14" role="img" aria-label="Schematic axial knee MRI slice">
            <rect width="64" height="64" fill="#0c1118" />
            <circle cx="32" cy="32" r="26" fill="none" stroke="#3a4656" strokeWidth="2" />
            <circle cx="32" cy="32" r="17" fill="none" stroke="#4b5a6e" strokeWidth="1.5" />
            <ellipse cx="26" cy="30" rx="6" ry="8" fill="#5c6d82" opacity="0.6" />
            <ellipse cx="40" cy="34" rx="5" ry="7" fill="#5c6d82" opacity="0.5" />
          </svg>
          <p className="text-center text-2xs leading-snug text-ink-faint">
            2D axial series
            <br />
            (schematic)
          </p>
        </div>

        <StageArrow />

        {/* 2. ResNet18 feature extraction */}
        <div className="flex flex-col items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4">
          <Boxes className="h-4 w-4 text-accent" aria-hidden="true" />
          <div className="flex h-14 items-end gap-[3px]" role="img" aria-label="Illustrative 512-dim embedding">
            {[0.4, 0.9, 0.3, 0.7, 0.5, 0.85, 0.2, 0.6].map((h, i) => (
              <div key={i} className="w-1.5 rounded-t-sm bg-accent/60" style={{ height: `${h * 100}%` }} />
            ))}
          </div>
          <p className="text-center text-2xs leading-snug text-ink-faint">
            ResNet18
            <br />
            512-dim embedding
          </p>
        </div>

        <StageArrow />

        {/* 3. Compression */}
        <div className="flex flex-col items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4">
          <Minimize2 className="h-4 w-4 text-accent" aria-hidden="true" />
          <div className="flex h-14 items-center gap-1.5" role="img" aria-label="Compressed to 4 scalars">
            {[0.72, 3.98, 1.55, 5.21].map((angle, i) => (
              <div
                key={i}
                className="flex h-9 w-9 items-center justify-center rounded-full border border-accent/40 bg-accent/10 font-mono text-[9px] text-accent"
              >
                {angle.toFixed(1)}
              </div>
            ))}
          </div>
          <p className="text-center text-2xs leading-snug text-ink-faint">
            PCA → 4 scalars
            <br />
            scaled to [0, 2π]
          </p>
        </div>

        <StageArrow />

        {/* 4. 4-qubit VQC */}
        <div className="flex flex-col items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4">
          <Atom className="h-4 w-4 text-accent" aria-hidden="true" />
          <div className="grid h-14 w-full grid-cols-4 gap-1" role="img" aria-label="Illustrative Pauli-Z expectations">
            {ILLUSTRATIVE_EXPECTATIONS.map((v, i) => (
              <div key={i} className="flex flex-col items-center justify-end gap-1">
                <span className="font-mono text-[8px] text-ink-faint">
                  q{i}
                </span>
                <div className="relative h-8 w-2 overflow-hidden rounded-full bg-surface-3">
                  <div
                    className="absolute bottom-1/2 w-full bg-accent"
                    style={{ height: `${Math.abs(v) * 50}%`, [v >= 0 ? "bottom" : "top"]: "50%" }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="text-center text-2xs leading-snug text-ink-faint">
            4-qubit VQC
            <br />3 variational layers
          </p>
        </div>

        <StageArrow />

        {/* 5. Risk output */}
        <div className="flex flex-col items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4">
          <div className="scale-[0.6] origin-center">
            <Gauge value={0.34} severity="Indeterminate" />
          </div>
          <p className="text-center text-2xs leading-snug text-ink-faint">Risk score</p>
        </div>

        <StageArrow />

        {/* 6. Explanation */}
        <div className="flex flex-col items-center gap-2 rounded-md border border-surface-3 bg-surface-0 px-3 py-4">
          <Eye className="h-4 w-4 text-accent" aria-hidden="true" />
          <svg viewBox="0 0 64 64" className="h-14 w-14" role="img" aria-label="Illustrative Grad-CAM attention overlay">
            <rect width="64" height="64" fill="#0c1118" />
            <circle cx="32" cy="32" r="26" fill="none" stroke="#3a4656" strokeWidth="2" />
            <defs>
              <radialGradient id="gradcam-swatch" cx="60%" cy="35%" r="55%">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.95" />
                <stop offset="45%" stopColor="#facc15" stopOpacity="0.55" />
                <stop offset="100%" stopColor="#facc15" stopOpacity="0" />
              </radialGradient>
            </defs>
            <circle cx="40" cy="26" r="18" fill="url(#gradcam-swatch)" />
          </svg>
          <p className="text-center text-2xs leading-snug text-ink-faint">Grad-CAM overlay</p>
        </div>
      </div>
      </div>
    </Surface>
  );
}
