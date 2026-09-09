import { Surface } from "@/components/ui/Surface";
import { Gauge } from "@/components/ui/Gauge";

/**
 * Schematic, not stock photography and not a real patient scan -- the
 * repo ships no sample DICOM asset for the marketing page to borrow, and
 * fabricating one and passing it off as a real study would be worse than
 * showing none. Every panel is labeled "schematic" so nobody mistakes it
 * for a real case. The MRI panel and the Grad-CAM panel are two distinct
 * SVGs (grayscale rings vs. a warm radial hotspot) -- never the same
 * image reused as its own heatmap, which the brief explicitly calls out.
 * Copy matches `ExplanationPanel.tsx`'s standing rule: Grad-CAM is
 * attention, not a segmentation or a diagnosis.
 */
function SchematicSlice() {
  return (
    <svg viewBox="0 0 200 200" className="h-full w-full" role="img" aria-label="Schematic axial knee MRI slice">
      <rect width="200" height="200" fill="#0a0e14" />
      <circle cx="100" cy="100" r="82" fill="none" stroke="#2a3442" strokeWidth="3" />
      <circle cx="100" cy="100" r="58" fill="none" stroke="#3a4656" strokeWidth="2" />
      <ellipse cx="78" cy="94" rx="20" ry="28" fill="#4b5a6e" opacity="0.55" />
      <ellipse cx="126" cy="106" rx="17" ry="24" fill="#4b5a6e" opacity="0.45" />
      <path d="M60 130 Q100 150 140 130" stroke="#5c6d82" strokeWidth="2" fill="none" opacity="0.6" />
    </svg>
  );
}

function SchematicGradCam() {
  return (
    <svg viewBox="0 0 200 200" className="h-full w-full" role="img" aria-label="Schematic Grad-CAM attention overlay">
      <rect width="200" height="200" fill="#0a0e14" />
      <circle cx="100" cy="100" r="82" fill="none" stroke="#2a3442" strokeWidth="3" />
      <defs>
        <radialGradient id="explain-hot" cx="65%" cy="45%" r="42%">
          <stop offset="0%" stopColor="#ef4444" stopOpacity="0.9" />
          <stop offset="40%" stopColor="#f97316" stopOpacity="0.6" />
          <stop offset="75%" stopColor="#facc15" stopOpacity="0.25" />
          <stop offset="100%" stopColor="#facc15" stopOpacity="0" />
        </radialGradient>
      </defs>
      <circle cx="128" cy="92" r="55" fill="url(#explain-hot)" />
    </svg>
  );
}

export function ExplainabilitySection() {
  return (
    <section className="border-t border-surface-3">
      <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6 sm:py-20">
        <p className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-accent">Explainability</p>
        <h2 className="mt-3 text-2xl font-bold tracking-tight text-ink-primary sm:text-3xl">
          Every score ships with an attention map
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-ink-muted">
          Grad-CAM is a model-attention visualization — which regions most influenced the ResNet18 embedding — not a
          segmentation and not a lesion detector. It does not confirm a finding; independent radiologist review is
          required for every result.
        </p>

        <div className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <Surface className="overflow-hidden">
            <div className="border-b border-surface-3 px-4 py-2.5 text-2xs font-semibold uppercase tracking-wide text-ink-faint">
              Input slice (schematic)
            </div>
            <div className="aspect-square p-6">
              <SchematicSlice />
            </div>
          </Surface>

          <Surface className="overflow-hidden">
            <div className="border-b border-surface-3 px-4 py-2.5 text-2xs font-semibold uppercase tracking-wide text-ink-faint">
              Grad-CAM overlay (schematic)
            </div>
            <div className="aspect-square p-6">
              <SchematicGradCam />
            </div>
          </Surface>

          <Surface className="flex flex-col overflow-hidden">
            <div className="border-b border-surface-3 px-4 py-2.5 text-2xs font-semibold uppercase tracking-wide text-ink-faint">
              Prediction (illustrative)
            </div>
            <div className="flex flex-1 flex-col items-center justify-center gap-3 p-6">
              <Gauge value={0.34} severity="Indeterminate" />
              <p className="text-center text-2xs leading-relaxed text-ink-faint">
                Risk probability + severity band. Not a confirmed diagnosis.
              </p>
            </div>
          </Surface>
        </div>
      </div>
    </section>
  );
}
