"use client";

import { useEffect, useState } from "react";
import { Layers, Upload } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
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

const WINDOWS = [
  { id: "soft-tissue", label: "Soft Tissue", wl: 40, ww: 400 },
  { id: "bone", label: "Bone", wl: 500, ww: 2000 },
] as const;

export function MriViewport({
  result,
  onUpload,
}: {
  result: DiagnosticResult | null;
  onUpload: (file: File) => void;
}) {
  const volume = result?.volume ?? null;

  const [requestedPlane, setRequestedPlane] = useState<AnatomicalPlane>("axial");
  const [slice, setSlice] = useState(0);
  const [heatmapOn, setHeatmapOn] = useState(true);
  const [opacity, setOpacity] = useState(65);
  const [windowPreset, setWindowPreset] = useState<(typeof WINDOWS)[number]["id"]>("soft-tissue");

  // A new result can change which planes/slices actually exist -- reset to
  // that result's own primary plane/slice rather than carrying over a
  // selection the new volume may not support.
  useEffect(() => {
    if (!volume) return;
    setRequestedPlane(volume.primaryPlane);
    setSlice(volume.primarySliceIndex);
  }, [result]);

  const filter = windowPreset === "bone" ? "contrast(1.4) brightness(1.1)" : "contrast(1.05) brightness(1)";

  const plane = volume ? resolvePlane(volume, requestedPlane) : null;
  const numSlices = plane && volume ? volume.planes[plane].numSlices : 0;
  const sliceIndex = plane && volume ? clampSliceIndex(volume, plane, slice) : 0;
  const baseImageSrc = plane && volume ? sliceImageSrc(volume, plane, sliceIndex) : null;
  const gradcamSrc =
    plane && volume && volume.gradcamOverlay && isGradcamVisible(volume, plane, sliceIndex)
      ? volume.gradcamOverlay
      : null;

  function selectPlane(next: AnatomicalPlane) {
    setRequestedPlane(next);
    if (volume) setSlice(clampSliceIndex(volume, next, slice));
  }

  return (
    <Card className="flex flex-col overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-3">
        <div className="flex gap-1 rounded-lg bg-slate-950 p-1">
          {PLANE_DISPLAY_ORDER.map((p) => {
            const available = volume?.planes[p]?.available ?? false;
            const isActive = plane === p;
            return (
              <button
                key={p}
                onClick={() => available && selectPlane(p)}
                disabled={!available}
                title={available ? undefined : "Unavailable — the backend did not produce this plane for this upload"}
                className={cn(
                  "rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                  !available && "cursor-not-allowed text-slate-700 opacity-50",
                  available && isActive && "bg-teal-500 text-slate-950",
                  available && !isActive && "text-slate-400 hover:text-slate-200",
                )}
              >
                {PLANE_LABELS[p]}
              </button>
            );
          })}
        </div>

        <label className="flex cursor-pointer items-center gap-1.5 rounded-md border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800">
          <Upload className="h-3.5 w-3.5" />
          Upload .dcm / .npy
          <input
            type="file"
            accept=".dcm,.dicom,.npy"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onUpload(file);
            }}
          />
        </label>
      </div>

      <div className="relative flex aspect-square items-center justify-center bg-black">
        <div
          className="relative h-full w-full bg-slate-900"
          style={{
            backgroundImage:
              "radial-gradient(circle at 50% 50%, rgba(148,163,184,0.14), transparent 65%)",
            filter,
          }}
        >
          {baseImageSrc ? (
            <>
              <img
                src={baseImageSrc}
                alt="MRI slice"
                className="absolute inset-0 h-full w-full object-cover opacity-90"
              />
              {heatmapOn && gradcamSrc && (
                <img
                  src={gradcamSrc}
                  alt="Grad-CAM overlay"
                  className="absolute inset-0 h-full w-full object-cover"
                  style={{ opacity: opacity / 100 }}
                />
              )}
            </>
          ) : (
            <div className="flex h-full w-full items-center justify-center px-6 text-center text-xs text-slate-600">
              {result
                ? "MRI slice unavailable for this result"
                : "Select a preset case or upload a scan"}
            </div>
          )}
          <div className="animate-scan-line pointer-events-none absolute inset-x-0 h-px bg-gradient-to-r from-transparent via-teal-400/70 to-transparent" />
        </div>

        {plane && (
          <div className="absolute left-3 top-3 rounded bg-slate-950/80 px-2 py-1 font-mono text-[10px] text-teal-400">
            {sliceCaption(plane, sliceIndex, numSlices)}
          </div>
        )}
      </div>

      <div className="space-y-4 border-t border-slate-800 px-4 py-4">
        <div>
          <div className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
            <span>Slice index</span>
            <span className="font-mono text-slate-300">
              {numSlices > 0 ? sliceIndex + 1 : 0} / {numSlices}
            </span>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(numSlices - 1, 0)}
            value={sliceIndex}
            disabled={numSlices <= 1}
            onChange={(e) => setSlice(Number(e.target.value))}
            className="w-full accent-teal-400 disabled:opacity-40"
          />
        </div>

        <div className="flex items-center justify-between">
          <label className="flex items-center gap-2 text-xs text-slate-400">
            <Layers className="h-3.5 w-3.5" />
            Grad-CAM Overlay
            {plane && volume?.gradcamOverlay && !isGradcamVisible(volume, plane, sliceIndex) && (
              <span className="text-slate-600">
                (only for {volume.gradcamPlane ? PLANE_LABELS[volume.gradcamPlane] : "?"} slice{" "}
                {(volume.gradcamSliceIndex ?? 0) + 1})
              </span>
            )}
          </label>
          <button
            onClick={() => setHeatmapOn((v) => !v)}
            disabled={!gradcamSrc}
            className={cn(
              "relative h-5 w-9 rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              heatmapOn && gradcamSrc ? "bg-teal-500" : "bg-slate-700",
            )}
          >
            <span
              className={cn(
                "absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform",
                heatmapOn && gradcamSrc ? "translate-x-4" : "translate-x-0.5",
              )}
            />
          </button>
        </div>

        {heatmapOn && gradcamSrc && (
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
              <span>Heatmap opacity</span>
              <span className="font-mono text-slate-300">{opacity}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={100}
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-full accent-teal-400"
            />
          </div>
        )}

        <div>
          <div className="mb-1.5 text-xs text-slate-400">Window / Level</div>
          <div className="flex gap-2">
            {WINDOWS.map((w) => (
              <button
                key={w.id}
                onClick={() => setWindowPreset(w.id)}
                className={cn(
                  "flex-1 rounded-md border px-3 py-1.5 text-xs font-medium transition-colors",
                  windowPreset === w.id
                    ? "border-teal-500 bg-teal-500/10 text-teal-400"
                    : "border-slate-700 text-slate-400 hover:bg-slate-800",
                )}
              >
                {w.label}
                <span className="ml-1 font-mono text-[10px] text-slate-600">
                  WL{w.wl}/WW{w.ww}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}
