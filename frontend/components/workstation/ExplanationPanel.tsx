import { PLANE_LABELS, sliceImageSrc, toImageSrc } from "@/lib/viewer";
import type { DiagnosticResult } from "@/lib/types";

/**
 * States what the Grad-CAM overlay is and is not. Never claims the
 * heatmap "proves" a finding, and never claims causality — Grad-CAM is a
 * model-attention visualization (which regions most influenced the
 * embedding), not a segmentation, not a lesion detector, and not evidence
 * that a highlighted region caused the score.
 *
 * The thumbnail below composites the SAME two assets the center viewer
 * uses (`sliceImageSrc` for the base slice, `volume.gradcamOverlay` for
 * the heatmap) — never a single image rendered twice and never a
 * re-generated/duplicate asset standing in for either half.
 */
export function ExplanationPanel({ result }: { result: DiagnosticResult | null }) {
  const volume = result?.volume ?? null;
  const hasOverlay = Boolean(volume?.gradcamOverlay && volume?.gradcamPlane !== null);

  const baseAtGradcamSlice =
    hasOverlay && volume && volume.gradcamPlane
      ? sliceImageSrc(volume, volume.gradcamPlane, volume.gradcamSliceIndex ?? 0)
      : null;

  return (
    <div className="space-y-3 text-xs text-ink-muted">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Why the model predicted this</h3>

      {hasOverlay && volume && result && volume.gradcamOverlay ? (
        <>
          <div className="relative aspect-square w-full max-w-[180px] overflow-hidden rounded-sm border border-surface-3 bg-black">
            {baseAtGradcamSlice && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={baseAtGradcamSlice} alt="" aria-hidden="true" className="absolute inset-0 h-full w-full object-contain" />
            )}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={toImageSrc(volume.gradcamOverlay)}
              alt="Grad-CAM model-attention overlay"
              className="absolute inset-0 h-full w-full object-contain"
              style={{ opacity: 0.7 }}
            />
          </div>

          <dl className="grid grid-cols-2 gap-x-3 gap-y-1">
            <dt className="text-ink-faint">Target</dt>
            <dd className="text-right text-ink-primary">{result.diagnosis}</dd>
            <dt className="text-ink-faint">Explained slice</dt>
            <dd className="text-right font-mono text-ink-primary">{(volume.gradcamSliceIndex ?? 0) + 1}</dd>
          </dl>

          <p>
            Computed for the {volume.gradcamPlane ? PLANE_LABELS[volume.gradcamPlane] : "—"} plane, one
            representative slice only — not every slice in this study.
          </p>
          <p className="text-ink-faint">
            Grad-CAM shows correlation between image regions and the model&apos;s output, not proof that a region
            caused the prediction. It is not a segmentation and does not confirm a lesion — independent radiologist
            review is required.
          </p>
        </>
      ) : (
        <p className="font-medium text-ink-primary">Explanation unavailable</p>
      )}
    </div>
  );
}
