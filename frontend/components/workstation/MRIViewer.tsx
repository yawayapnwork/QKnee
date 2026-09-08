"use client";

import { useEffect, useState } from "react";
import { Layers } from "lucide-react";
import { Tabs } from "@/components/ui/Tabs";
import { Switch } from "@/components/ui/Switch";
import { EmptyState } from "@/components/shared/EmptyState";
import {
  PLANE_DISPLAY_ORDER,
  PLANE_LABELS,
  clampSliceIndex,
  isGradcamVisible,
  resolvePlane,
  sliceCaption,
  sliceImageSrc,
} from "@/lib/viewer";
import type { AnatomicalPlane, DiagnosticResult } from "@/lib/types";

/**
 * CENTER zone: the imaging surface. Every control here either changes what
 * is rendered, or does not exist -- per FRONTEND_REDESIGN.md Part "MRI
 * viewer UX" and the execution mandate's rule 10/11: no fabricated
 * Window/Level values, no permanent decorative animation, no claim of a
 * plane/slice count the backend didn't actually report.
 *
 * `lib/viewer.ts` is unmodified data-layer logic (audited, correct):
 * `resolvePlane` never invents an unavailable plane, `sliceImageSrc`
 * returns `null` rather than a substitute image, and the base slice
 * (`sliceImageSrc`) and Grad-CAM overlay (`volume.gradcamOverlay`) are
 * always two distinct assets -- this component never blends them into one
 * image or lets one stand in for the other.
 */
export function MRIViewer({ result }: { result: DiagnosticResult | null }) {
  const volume = result?.volume ?? null;

  const [requestedPlane, setRequestedPlane] = useState<AnatomicalPlane>("axial");
  const [slice, setSlice] = useState(0);
  const [heatmapOn, setHeatmapOn] = useState(true);
  const [opacity, setOpacity] = useState(65);

  useEffect(() => {
    if (!volume) return;
    setRequestedPlane(volume.primaryPlane);
    setSlice(volume.primarySliceIndex);
  }, [result, volume]);

  const plane = volume ? resolvePlane(volume, requestedPlane) : null;
  const numSlices = plane && volume ? volume.planes[plane].numSlices : 0;
  const sliceIndex = plane && volume ? clampSliceIndex(volume, plane, slice) : 0;
  const baseImageSrc = plane && volume ? sliceImageSrc(volume, plane, sliceIndex) : null;
  const gradcamSrc =
    plane && volume && volume.gradcamOverlay && isGradcamVisible(volume, plane, sliceIndex)
      ? volume.gradcamOverlay
      : null;
  const gradcamExistsElsewhere = Boolean(plane && volume?.gradcamOverlay && !gradcamSrc);

  function selectPlane(next: AnatomicalPlane) {
    setRequestedPlane(next);
    if (volume) setSlice(clampSliceIndex(volume, next, slice));
  }

  const planeTabs = PLANE_DISPLAY_ORDER.map((p) => ({
    value: p,
    label: PLANE_LABELS[p],
    disabled: !(volume?.planes[p]?.available ?? false),
    disabledReason: "This plane was not produced for this study — a single-series MRI upload only has a real axial stack.",
  }));

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-surface-3 px-4 py-3">
        <Tabs items={planeTabs} value={plane ?? "axial"} onChange={selectPlane} label="Anatomical plane" />
      </div>

      <div className="relative flex flex-1 items-center justify-center bg-black">
        {baseImageSrc ? (
          <div className="relative h-full w-full">
            {/* eslint-disable-next-line @next/next/no-img-element -- base64 data URIs, not a static/optimizable asset */}
            <img src={baseImageSrc} alt="MRI slice" className="absolute inset-0 h-full w-full object-contain" />
            {heatmapOn && gradcamSrc && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={gradcamSrc}
                alt="Grad-CAM model-attention overlay"
                className="absolute inset-0 h-full w-full object-contain"
                style={{ opacity: opacity / 100 }}
              />
            )}
            {plane && (
              <div className="absolute left-3 top-3 rounded bg-surface-0/85 px-2 py-1 font-mono text-[11px] text-ink-primary">
                {sliceCaption(plane, sliceIndex, numSlices)}
              </div>
            )}
          </div>
        ) : (
          <EmptyState
            title={result ? "MRI slice unavailable" : "No study loaded"}
            description={
              result
                ? "This result has no viewable slice data for the requested plane."
                : "Select a demo case or upload a scan to begin."
            }
          />
        )}
      </div>

      <div className="space-y-4 border-t border-surface-3 px-4 py-4">
        <div>
          <label htmlFor="slice-slider" className="mb-1.5 flex items-center justify-between text-xs text-ink-muted">
            <span>Slice</span>
            <span className="font-mono text-ink-primary">
              {numSlices > 0 ? sliceIndex + 1 : 0} / {numSlices}
            </span>
          </label>
          <input
            id="slice-slider"
            type="range"
            min={0}
            max={Math.max(numSlices - 1, 0)}
            value={sliceIndex}
            disabled={numSlices <= 1}
            onChange={(e) => setSlice(Number(e.target.value))}
            aria-valuetext={`Slice ${numSlices > 0 ? sliceIndex + 1 : 0} of ${numSlices}`}
            className="w-full accent-accent disabled:opacity-40"
          />
        </div>

        <div className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-xs text-ink-muted">
            <Layers className="h-3.5 w-3.5" aria-hidden="true" />
            Grad-CAM overlay
            {gradcamExistsElsewhere && plane && volume && (
              <span className="text-ink-faint">
                (only computed for {volume.gradcamPlane ? PLANE_LABELS[volume.gradcamPlane] : "?"} slice{" "}
                {(volume.gradcamSliceIndex ?? 0) + 1})
              </span>
            )}
          </span>
          <Switch
            checked={heatmapOn && Boolean(gradcamSrc)}
            onChange={() => setHeatmapOn((v) => !v)}
            disabled={!gradcamSrc}
            label="Toggle Grad-CAM overlay"
          />
        </div>

        {heatmapOn && gradcamSrc && (
          <div>
            <label htmlFor="opacity-slider" className="mb-1.5 flex items-center justify-between text-xs text-ink-muted">
              <span>Overlay opacity</span>
              <span className="font-mono text-ink-primary">{opacity}%</span>
            </label>
            <input
              id="opacity-slider"
              type="range"
              min={0}
              max={100}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              aria-valuetext={`${opacity} percent`}
              className="w-full accent-accent"
            />
          </div>
        )}
      </div>
    </div>
  );
}
