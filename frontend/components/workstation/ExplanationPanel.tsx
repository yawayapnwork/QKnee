import { PLANE_LABELS } from "@/lib/viewer";
import type { VolumeView } from "@/lib/types";

/**
 * States what the Grad-CAM overlay is and is not. Never claims the
 * heatmap "proves" a finding -- Grad-CAM is a model-attention
 * visualization (which regions of the image most influenced the
 * embedding), not a segmentation or a lesion detector.
 */
export function ExplanationPanel({ volume }: { volume: VolumeView | null }) {
  const hasOverlay = Boolean(volume?.gradcamOverlay);

  return (
    <div className="space-y-2 text-xs text-ink-muted">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Explanation</h3>
      {hasOverlay && volume ? (
        <>
          <p>
            Model attention visualization (Grad-CAM), computed for{" "}
            <span className="font-mono text-ink-primary">
              {volume.gradcamPlane ? PLANE_LABELS[volume.gradcamPlane] : "—"} slice{" "}
              {(volume.gradcamSliceIndex ?? 0) + 1}
            </span>{" "}
            only — not every slice in this study.
          </p>
          <p className="text-ink-faint">
            This overlay highlights regions that most influenced the model&apos;s embedding. It is not a segmentation
            and does not confirm a lesion — independent radiologist review is required.
          </p>
        </>
      ) : (
        <p>No Grad-CAM explanation is available for this result.</p>
      )}
    </div>
  );
}
