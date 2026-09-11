import type { AnatomicalPlane, PlaneInfo, PredictionResponse, RawPlaneInfo, VolumeView } from "./types";

export const PLANE_LABELS: Record<AnatomicalPlane, string> = {
  axial: "Axial",
  coronal: "Coronal",
  sagittal: "Sagittal",
};

/** Display order for the plane selector — unrelated to the `AnatomicalPlane` union's declaration order. */
export const PLANE_DISPLAY_ORDER: AnatomicalPlane[] = ["sagittal", "coronal", "axial"];

function toPlaneInfo(raw: RawPlaneInfo | undefined): PlaneInfo {
  if (!raw) return { available: false, numSlices: 0, slices: [] };
  return { available: raw.available, numSlices: raw.num_slices, slices: raw.slices };
}

/**
 * Builds this result's `VolumeView` **only** from the backend's own
 * `/predict` response — mirrors `quantum-telemetry.ts`'s single-call-site
 * pattern for the same reason: it takes no preset argument, so preset/demo
 * volume data cannot leak into a live result.
 */
export function volumeViewFromPrediction(prediction: PredictionResponse): VolumeView {
  return {
    planes: {
      axial: toPlaneInfo(prediction.planes?.axial),
      coronal: toPlaneInfo(prediction.planes?.coronal),
      sagittal: toPlaneInfo(prediction.planes?.sagittal),
    },
    primaryPlane: prediction.primary_plane,
    primarySliceIndex: prediction.primary_slice_index,
    gradcamOverlay: prediction.gradcam_overlay,
    gradcamPlane: prediction.gradcam_plane,
    gradcamSliceIndex: prediction.gradcam_slice_index,
    gradcamDegenerate: prediction.gradcam_degenerate,
  };
}

// `placeholderImageDataUri`/`volumeViewFromPreset` (synthetic gradient
// images standing in for a preset's non-existent MRI data) are deleted --
// demo cases are now real RSNA Knee studies scored once offline (see
// `scripts/build_real_demo_cases.py`), with real slice images, and served
// via `GET /api/cases/{id}` as an ordinary `PredictionResponse`, so they go
// through `volumeViewFromPrediction` above like any other result.

/** Normalizes a raw base64 PNG or an already-prefixed data URI into a `<img src>`-ready data URI. */
export function toImageSrc(base64OrDataUri: string): string {
  return base64OrDataUri.startsWith("data:") ? base64OrDataUri : `data:image/png;base64,${base64OrDataUri}`;
}

/**
 * Resolves a requested plane to one the volume can actually display:
 * the request itself, if available; otherwise `primaryPlane` (always
 * assumed available — it's what was actually classified); otherwise the
 * first available plane; otherwise `null` if nothing is available at all
 * (e.g. `CachedFallbackBackend` responses, which have no separate raw
 * slice recorded for any plane). The UI must treat a `null` result as
 * "nothing to show", never fall back to a fabricated image.
 */
export function resolvePlane(volume: VolumeView, requested: AnatomicalPlane): AnatomicalPlane | null {
  if (volume.planes[requested]?.available) return requested;
  if (volume.planes[volume.primaryPlane]?.available) return volume.primaryPlane;
  const firstAvailable = PLANE_DISPLAY_ORDER.find((plane) => volume.planes[plane]?.available);
  return firstAvailable ?? null;
}

/** Clamps `index` into `[0, numSlices(plane) - 1]` (or `0` if the plane has no slices at all). */
export function clampSliceIndex(volume: VolumeView, plane: AnatomicalPlane, index: number): number {
  const numSlices = volume.planes[plane]?.numSlices ?? 0;
  if (numSlices <= 0) return 0;
  return Math.min(Math.max(index, 0), numSlices - 1);
}

/** The exact slice image for `plane`/`index`, or `null` if that plane/index has no real data. */
export function sliceImageSrc(volume: VolumeView, plane: AnatomicalPlane, index: number): string | null {
  const planeInfo = volume.planes[plane];
  if (!planeInfo?.available) return null;
  const slice = planeInfo.slices[index];
  return slice ? toImageSrc(slice) : null;
}

/**
 * Whether the Grad-CAM overlay applies to exactly this plane/slice — Grad-CAM
 * is computed for one representative slice only (see
 * `PredictionResponse.gradcam_slice_index`'s docstring), so it must never be
 * shown against any other slice as if every slice had one.
 */
export function isGradcamVisible(volume: VolumeView, plane: AnatomicalPlane, index: number): boolean {
  return volume.gradcamOverlay !== null && volume.gradcamPlane === plane && volume.gradcamSliceIndex === index;
}

/** "Plane: Axial · Slice: 15 / 32" — always from real `numSlices`, never a hardcoded count. */
export function sliceCaption(plane: AnatomicalPlane, index: number, numSlices: number): string {
  const displayIndex = numSlices > 0 ? index + 1 : 0;
  return `Plane: ${PLANE_LABELS[plane]} · Slice: ${displayIndex} / ${numSlices}`;
}
