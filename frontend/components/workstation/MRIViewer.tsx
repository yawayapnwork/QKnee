"use client";

import { useEffect, useRef, useState } from "react";
import { Layers, ZoomIn, ZoomOut, RotateCcw, Sun, Contrast } from "lucide-react";
import { Tabs } from "@/components/ui/Tabs";
import { Switch } from "@/components/ui/Switch";
import { EmptyState } from "@/components/shared/EmptyState";
import {
  PLANE_DISPLAY_ORDER,
  PLANE_LABELS,
  clampSliceIndex,
  isGradcamVisible,
  resolvePlane,
  sliceImageSrc,
  toImageSrc,
} from "@/lib/viewer";
import { cn } from "@/lib/utils";
import type { AnatomicalPlane, DiagnosticResult } from "@/lib/types";

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.5;
const PAN_STEP = 24;

/**
 * CENTER zone: the imaging surface, deliberately the dominant region of
 * the screen (the flex-1 image area below the toolbar/plane strip
 * consumes whatever height the 3-zone grid gives it). Every control here
 * either changes what is rendered, or does not exist — no fabricated
 * Window/Level values, no permanent decorative animation, no claim of a
 * plane/slice count the backend didn't actually report.
 *
 * Zoom/pan/window-level are pure client-side VIEWER transforms (CSS
 * transform + filter) applied on top of the real slice image — they never
 * alter, re-render, or recompute the underlying pixel data, and "Window /
 * Level" here is a brightness/contrast filter on the already-rendered PNG,
 * not a recomputation from raw intensities. That's stated in the control's
 * own label rather than left ambiguous.
 *
 * `lib/viewer.ts` is unmodified data-layer logic (audited, tested):
 * `resolvePlane` never invents an unavailable plane, `sliceImageSrc`
 * returns `null` rather than a substitute image, and the base slice and
 * Grad-CAM overlay are always two distinct assets — this component never
 * blends them into one image or lets one stand in for the other.
 */
export function MRIViewer({ result }: { result: DiagnosticResult | null }) {
  const volume = result?.volume ?? null;

  const [requestedPlane, setRequestedPlane] = useState<AnatomicalPlane>("axial");
  const [slice, setSlice] = useState(0);
  const [heatmapOn, setHeatmapOn] = useState(true);
  const [opacity, setOpacity] = useState(65);

  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);

  const dragState = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  // Pointer `move` can fire well over 60 times/second while dragging; only
  // the latest position matters, so each move overwrites a pending
  // coordinate rather than queuing a `setPan` per event, and the actual
  // state update (the expensive part -- it re-renders this component and
  // recalculates the CSS transform) happens at most once per animation
  // frame via `requestAnimationFrame`.
  const pendingPan = useRef<{ x: number; y: number } | null>(null);
  const rafId = useRef<number | null>(null);

  useEffect(() => {
    if (!volume) return;
    setRequestedPlane(volume.primaryPlane);
    setSlice(volume.primarySliceIndex);
    resetView();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, volume]);

  useEffect(() => {
    return () => {
      if (rafId.current !== null) cancelAnimationFrame(rafId.current);
    };
  }, []);

  const plane = volume ? resolvePlane(volume, requestedPlane) : null;
  const numSlices = plane && volume ? volume.planes[plane].numSlices : 0;
  const sliceIndex = plane && volume ? clampSliceIndex(volume, plane, slice) : 0;
  const baseImageSrc = plane && volume ? sliceImageSrc(volume, plane, sliceIndex) : null;
  const gradcamSrc =
    plane && volume && volume.gradcamOverlay && isGradcamVisible(volume, plane, sliceIndex)
      ? toImageSrc(volume.gradcamOverlay)
      : null;
  const gradcamExistsElsewhere = Boolean(plane && volume?.gradcamOverlay && !gradcamSrc);

  function selectPlane(next: AnatomicalPlane) {
    setRequestedPlane(next);
    if (volume) setSlice(clampSliceIndex(volume, next, slice));
    resetView();
  }

  function resetView() {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setBrightness(100);
    setContrast(100);
  }

  function applyZoom(next: number) {
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next));
    setZoom(clamped);
    if (clamped === MIN_ZOOM) setPan({ x: 0, y: 0 });
  }

  function handlePointerDown(e: React.PointerEvent<HTMLDivElement>) {
    if (zoom <= MIN_ZOOM) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    dragState.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
    setIsDragging(true);
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragState.current) return;
    const dx = e.clientX - dragState.current.startX;
    const dy = e.clientY - dragState.current.startY;
    pendingPan.current = { x: dragState.current.panX + dx, y: dragState.current.panY + dy };
    if (rafId.current !== null) return;
    rafId.current = requestAnimationFrame(() => {
      rafId.current = null;
      if (pendingPan.current) setPan(pendingPan.current);
    });
  }

  function stopDrag() {
    dragState.current = null;
    setIsDragging(false);
    if (rafId.current !== null) {
      cancelAnimationFrame(rafId.current);
      rafId.current = null;
    }
    // Flush the last pending position so the image doesn't visibly settle
    // one frame short of where the pointer actually stopped.
    if (pendingPan.current) {
      setPan(pendingPan.current);
      pendingPan.current = null;
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key === "+" || e.key === "=") applyZoom(zoom + ZOOM_STEP);
    else if (e.key === "-" || e.key === "_") applyZoom(zoom - ZOOM_STEP);
    else if (e.key === "0") resetView();
    else if (zoom > MIN_ZOOM && e.key.startsWith("Arrow")) {
      e.preventDefault();
      setPan((p) => ({
        x: p.x + (e.key === "ArrowLeft" ? PAN_STEP : e.key === "ArrowRight" ? -PAN_STEP : 0),
        y: p.y + (e.key === "ArrowUp" ? PAN_STEP : e.key === "ArrowDown" ? -PAN_STEP : 0),
      }));
    }
  }

  const planeTabs = PLANE_DISPLAY_ORDER.map((p) => ({
    value: p,
    label: PLANE_LABELS[p],
    disabled: !(volume?.planes[p]?.available ?? false),
    disabledReason: "This plane was not produced for this study — a single-series MRI upload only has a real axial stack.",
  }));

  const hasUnavailablePlane = planeTabs.some((tab) => tab.disabled);
  const viewAdjusted = zoom !== 1 || pan.x !== 0 || pan.y !== 0 || brightness !== 100 || contrast !== 100;

  return (
    <div className="flex h-full flex-col min-h-0">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-surface-3 bg-white px-4 py-2.5">
        <Tabs items={planeTabs} value={plane ?? "axial"} onChange={selectPlane} label="Anatomical plane" />
        {/* Stated inline, not hover-only -- a disabled tab isn't keyboard-
            focusable at all, so its `title` tooltip is unreachable without
            a mouse. The reason must be visible unconditionally. */}
        {hasUnavailablePlane && (
          <span className="text-2xs text-ink-faint">{planeTabs.find((t) => t.disabled)?.disabledReason}</span>
        )}
      </div>

      {baseImageSrc && (
        <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-surface-3 bg-white px-3 py-1.5" role="toolbar" aria-label="Viewer controls">
          <ToolButton label="Zoom out" onClick={() => applyZoom(zoom - ZOOM_STEP)} disabled={zoom <= MIN_ZOOM}>
            <ZoomOut className="h-3.5 w-3.5" aria-hidden="true" />
          </ToolButton>
          <span className="w-12 text-center font-mono text-2xs text-ink-muted" aria-hidden="true">
            {Math.round(zoom * 100)}%
          </span>
          <ToolButton label="Zoom in" onClick={() => applyZoom(zoom + ZOOM_STEP)} disabled={zoom >= MAX_ZOOM}>
            <ZoomIn className="h-3.5 w-3.5" aria-hidden="true" />
          </ToolButton>

          <div className="mx-1.5 h-4 w-px bg-surface-3" aria-hidden="true" />

          <label className="flex items-center gap-1.5 text-2xs text-ink-muted">
            <Sun className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="sr-only">Window / Level — brightness</span>
            <input
              type="range"
              min={40}
              max={160}
              value={brightness}
              onChange={(e) => setBrightness(Number(e.target.value))}
              aria-label="Window / Level — brightness"
              className="w-16 accent-accent"
            />
          </label>
          <label className="flex items-center gap-1.5 text-2xs text-ink-muted">
            <Contrast className="h-3.5 w-3.5" aria-hidden="true" />
            <span className="sr-only">Window / Level — contrast</span>
            <input
              type="range"
              min={40}
              max={200}
              value={contrast}
              onChange={(e) => setContrast(Number(e.target.value))}
              aria-label="Window / Level — contrast"
              className="w-16 accent-accent"
            />
          </label>

          <div className="mx-1.5 h-4 w-px bg-surface-3" aria-hidden="true" />

          <ToolButton label="Reset view" onClick={resetView} disabled={!viewAdjusted}>
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
          </ToolButton>
        </div>
      )}

      <div className="relative flex flex-1 min-h-0 items-center justify-center overflow-hidden bg-black">
        {baseImageSrc ? (
          <div
            role="img"
            aria-label={`MRI slice, ${PLANE_LABELS[plane as AnatomicalPlane]} plane, slice ${sliceIndex + 1} of ${numSlices}. Use plus/minus to zoom, arrow keys to pan when zoomed, 0 to reset.`}
            tabIndex={0}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={stopDrag}
            onPointerLeave={stopDrag}
            onKeyDown={handleKeyDown}
            className={cn(
              "relative h-full w-full outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent",
              zoom > 1 && (isDragging ? "cursor-grabbing" : "cursor-grab"),
            )}
          >
            <div
              className="absolute inset-0 h-full w-full transition-transform duration-fast"
              style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element -- base64 data URIs, not a static/optimizable asset */}
              <img
                src={baseImageSrc}
                alt="MRI slice"
                draggable={false}
                className="absolute inset-0 h-full w-full object-contain"
                style={{ filter: `brightness(${brightness}%) contrast(${contrast}%)` }}
              />
              {heatmapOn && gradcamSrc && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={gradcamSrc}
                  alt="Grad-CAM model-attention overlay"
                  draggable={false}
                  className="absolute inset-0 h-full w-full object-contain"
                  style={{ opacity: opacity / 100 }}
                />
              )}
            </div>

            <div className="pointer-events-none absolute left-3 top-3 rounded-sm border border-white/10 bg-black/80 px-2 py-1 font-mono text-[11px] leading-tight text-white/90">
              <div>
                Slice {numSlices > 0 ? sliceIndex + 1 : 0} / {numSlices}
              </div>
              <div className="text-white/60">Plane: {plane ? PLANE_LABELS[plane] : "—"}</div>
            </div>
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

      <div className="shrink-0 space-y-4 border-t border-surface-3 bg-white px-4 py-4">
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

function ToolButton({
  label,
  onClick,
  disabled,
  children,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      className="rounded-sm p-1.5 text-ink-secondary transition-colors hover:bg-surface-2 hover:text-ink-primary disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent"
    >
      {children}
    </button>
  );
}
