"use client";

import { useState } from "react";
import { Microscope } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { ExplanationWorkspace } from "@/components/workstation/ExplanationWorkspace";
import { sliceImageSrc, toImageSrc } from "@/lib/viewer";
import type { DiagnosticResult } from "@/lib/types";

/**
 * Sidebar entry point into the full `ExplanationWorkspace` -- a small,
 * honest preview (the same base+overlay composite the workspace's Visual
 * Evidence section shows, never a re-generated duplicate) plus one action
 * that opens the five-section workspace. This panel itself makes no
 * claims about the overlay beyond what's visible; the substantive
 * "model attention, not proof of a lesion" language lives once, in the
 * workspace, rather than being duplicated (and risking drift) here too.
 */
export function ExplanationPanel({ result }: { result: DiagnosticResult | null }) {
  const [open, setOpen] = useState(false);
  const volume = result?.volume ?? null;
  const hasOverlay = Boolean(volume?.gradcamOverlay && volume?.gradcamPlane !== null);

  const baseAtGradcamSlice =
    hasOverlay && volume && volume.gradcamPlane
      ? sliceImageSrc(volume, volume.gradcamPlane, volume.gradcamSliceIndex ?? 0)
      : null;

  return (
    <div className="space-y-3 text-xs text-ink-muted">
      <h3 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-ink-faint">
        <Microscope className="h-3.5 w-3.5" aria-hidden="true" />
        Why the model predicted this
      </h3>

      {!result ? (
        <p className="font-medium text-ink-primary">Explanation unavailable</p>
      ) : (
        <>
          {hasOverlay && volume && volume.gradcamOverlay ? (
            <div className="relative aspect-square w-full max-w-[180px] overflow-hidden rounded-sm border border-surface-3 bg-black">
              {baseAtGradcamSlice && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={baseAtGradcamSlice} alt="" aria-hidden="true" className="absolute inset-0 h-full w-full object-contain" />
              )}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={toImageSrc(volume.gradcamOverlay)}
                alt="Grad-CAM model-attention overlay preview"
                className="absolute inset-0 h-full w-full object-contain"
                style={{ opacity: 0.7 }}
              />
            </div>
          ) : (
            <p className="text-2xs text-ink-faint">No Grad-CAM overlay for this result — model output and quantum contribution are still available.</p>
          )}

          <Button variant="secondary" size="sm" onClick={() => setOpen(true)} className="w-full">
            Open Explanation Workspace
          </Button>

          <ExplanationWorkspace open={open} onClose={() => setOpen(false)} result={result} />
        </>
      )}
    </div>
  );
}
