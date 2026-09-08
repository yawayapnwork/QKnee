import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  clampSliceIndex,
  isGradcamVisible,
  resolvePlane,
  sliceCaption,
  sliceImageSrc,
  toImageSrc,
  volumeViewFromPrediction,
  volumeViewFromPreset,
} from "../viewer";
import { PRESET_CASES } from "../mock-data";
import { livePrediction } from "./fixtures";

describe("AUDIT.md P0 #3/#4 regression: base MRI slice and Grad-CAM overlay are separate assets", () => {
  it("base image and gradcam overlay come from different fields, never the same asset", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    const baseSrc = sliceImageSrc(volume, "axial", volume.gradcamSliceIndex!);
    expect(baseSrc).not.toBeNull();
    expect(baseSrc).not.toEqual(toImageSrc(volume.gradcamOverlay!));
    expect(baseSrc).toContain("AXIAL_SLICE_12_BASE64");
    expect(volume.gradcamOverlay).toContain("AXIAL_SLICE_12_OVERLAY_BASE64");
  });

  it("preset/demo volumes also use two distinct generated images, never the same asset twice", () => {
    for (const preset of PRESET_CASES) {
      const volume = volumeViewFromPreset(preset);
      const baseSrc = sliceImageSrc(volume, "axial", 0);
      expect(baseSrc).not.toBeNull();
      expect(baseSrc).not.toEqual(volume.gradcamOverlay);
    }
  });

  it("gradcam overlay is null (never a substitute base image) when the backend computed none", () => {
    const volume = volumeViewFromPrediction(
      livePrediction({ gradcam_overlay: null, gradcam_plane: null, gradcam_slice_index: null }),
    );
    expect(volume.gradcamOverlay).toBeNull();
    expect(isGradcamVisible(volume, "axial", volume.primarySliceIndex)).toBe(false);
  });
});

describe("plane switching", () => {
  it("resolves to the requested plane when the backend reports it available", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    expect(resolvePlane(volume, "axial")).toBe("axial");
  });

  it("falls back to primaryPlane, then the first available plane, when the requested plane is unavailable", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    expect(resolvePlane(volume, "coronal")).toBe("axial");
    expect(resolvePlane(volume, "sagittal")).toBe("axial");
  });

  it("returns null when no plane at all is available (e.g. a cache-fallback response)", () => {
    const volume = volumeViewFromPrediction(
      livePrediction({
        planes: {
          axial: { available: false, num_slices: 0, slices: [] },
          coronal: { available: false, num_slices: 0, slices: [] },
          sagittal: { available: false, num_slices: 0, slices: [] },
        },
      }),
    );
    expect(resolvePlane(volume, "axial")).toBeNull();
    expect(sliceImageSrc(volume, "axial", 0)).toBeNull();
  });

  it("switching to a different, real plane actually changes which slice image is shown", () => {
    const volume = volumeViewFromPrediction(
      livePrediction({
        planes: {
          axial: { available: true, num_slices: 3, slices: ["AX_0", "AX_1", "AX_2"] },
          coronal: { available: true, num_slices: 5, slices: ["COR_0", "COR_1", "COR_2", "COR_3", "COR_4"] },
          sagittal: { available: false, num_slices: 0, slices: [] },
        },
      }),
    );
    const axialSrc = sliceImageSrc(volume, "axial", 1);
    const coronalSrc = sliceImageSrc(volume, "coronal", 1);
    expect(axialSrc).toContain("AX_1");
    expect(coronalSrc).toContain("COR_1");
    expect(axialSrc).not.toEqual(coronalSrc);
  });
});

describe("unavailable planes", () => {
  it("reports availability exactly as the backend sent it, per plane", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    expect(volume.planes.axial.available).toBe(true);
    expect(volume.planes.coronal.available).toBe(false);
    expect(volume.planes.sagittal.available).toBe(false);
  });

  it("an unavailable plane has zero slices and no fabricated image", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    expect(volume.planes.coronal.numSlices).toBe(0);
    expect(sliceImageSrc(volume, "coronal", 0)).toBeNull();
  });

  it("a preset/demo case only ever claims the axial plane, never Coronal/Sagittal", () => {
    const volume = volumeViewFromPreset(PRESET_CASES[0]);
    expect(volume.planes.axial.available).toBe(true);
    expect(volume.planes.coronal.available).toBe(false);
    expect(volume.planes.sagittal.available).toBe(false);
  });
});

describe("slice switching and bounds", () => {
  it("clamps a requested slice index into [0, numSlices - 1] for the given plane", () => {
    const volume = volumeViewFromPrediction(livePrediction()); // 24 axial slices
    expect(clampSliceIndex(volume, "axial", -5)).toBe(0);
    expect(clampSliceIndex(volume, "axial", 0)).toBe(0);
    expect(clampSliceIndex(volume, "axial", 23)).toBe(23);
    expect(clampSliceIndex(volume, "axial", 999)).toBe(23);
  });

  it("clamps to 0 for a plane with no slices at all", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    expect(clampSliceIndex(volume, "coronal", 5)).toBe(0);
  });

  it("correctly handles a single-slice input (numSlices === 1)", () => {
    const volume = volumeViewFromPrediction(
      livePrediction({
        planes: {
          axial: { available: true, num_slices: 1, slices: ["ONLY_SLICE"] },
          coronal: { available: false, num_slices: 0, slices: [] },
          sagittal: { available: false, num_slices: 0, slices: [] },
        },
        primary_slice_index: 0,
        gradcam_slice_index: 0,
      }),
    );
    expect(clampSliceIndex(volume, "axial", 0)).toBe(0);
    expect(clampSliceIndex(volume, "axial", 5)).toBe(0);
    expect(sliceImageSrc(volume, "axial", 0)).toContain("ONLY_SLICE");
    expect(sliceCaption("axial", 0, volume.planes.axial.numSlices)).toBe("Plane: Axial · Slice: 1 / 1");
  });

  it("changing the slice index actually changes the displayed image", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    const slice0 = sliceImageSrc(volume, "axial", 0);
    const slice10 = sliceImageSrc(volume, "axial", 10);
    expect(slice0).not.toEqual(slice10);
    expect(slice0).toContain("AXIAL_SLICE_0_BASE64");
    expect(slice10).toContain("AXIAL_SLICE_10_BASE64");
  });

  it("never hardcodes a slice count -- it always reflects the backend's own num_slices", () => {
    const volumeA = volumeViewFromPrediction(
      livePrediction({ planes: { axial: { available: true, num_slices: 7, slices: Array(7).fill("s") }, coronal: { available: false, num_slices: 0, slices: [] }, sagittal: { available: false, num_slices: 0, slices: [] } } }),
    );
    const volumeB = volumeViewFromPrediction(
      livePrediction({ planes: { axial: { available: true, num_slices: 40, slices: Array(40).fill("s") }, coronal: { available: false, num_slices: 0, slices: [] }, sagittal: { available: false, num_slices: 0, slices: [] } } }),
    );
    expect(volumeA.planes.axial.numSlices).toBe(7);
    expect(volumeB.planes.axial.numSlices).toBe(40);
  });
});

describe("display caption", () => {
  it('formats "Plane: <Name> · Slice: <1-based> / <total>"', () => {
    expect(sliceCaption("axial", 14, 32)).toBe("Plane: Axial · Slice: 15 / 32");
    expect(sliceCaption("coronal", 0, 10)).toBe("Plane: Coronal · Slice: 1 / 10");
    expect(sliceCaption("sagittal", 4, 0)).toBe("Plane: Sagittal · Slice: 0 / 0");
  });
});

describe("Grad-CAM exists for only one slice, and only overlay opacity affects it", () => {
  it("isGradcamVisible is true only for the exact plane/slice the backend recorded", () => {
    const volume = volumeViewFromPrediction(livePrediction()); // gradcam at axial/12
    expect(isGradcamVisible(volume, "axial", 12)).toBe(true);
    expect(isGradcamVisible(volume, "axial", 11)).toBe(false);
    expect(isGradcamVisible(volume, "axial", 13)).toBe(false);
  });

  it("does not pretend every slice has Grad-CAM: gradcamPlane/gradcamSliceIndex identify the one slice it belongs to", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    expect(volume.gradcamPlane).toBe("axial");
    expect(volume.gradcamSliceIndex).toBe(12);
  });

  it("the base image at the Grad-CAM slice is unaffected by opacity -- opacity is purely a caller-side compositing concern over the overlay asset", () => {
    const volume = volumeViewFromPrediction(livePrediction());
    const baseSrc = sliceImageSrc(volume, "axial", 12);
    // The base image string itself carries no opacity information; opacity
    // is applied by the caller only to the separate overlay asset.
    expect(baseSrc).toContain("AXIAL_SLICE_12_BASE64");
    expect(baseSrc).not.toContain("OVERLAY");
  });
});

describe("overlay opacity affects only the Grad-CAM layer, never the base image", () => {
  const MRI_VIEWPORT = join(__dirname, "..", "..", "components", "workstation", "MriViewport.tsx");
  const source = readFileSync(MRI_VIEWPORT, "utf-8");

  it("the base MRI <img> has no opacity style driven by the opacity slider state", () => {
    const baseImgBlock = source.slice(source.indexOf('alt="MRI slice"') - 200, source.indexOf('alt="MRI slice"') + 100);
    expect(baseImgBlock).not.toMatch(/opacity\s*\/\s*100/);
  });

  it("the Grad-CAM overlay <img> is the only element styled from the opacity slider state", () => {
    const overlayImgBlock = source.slice(
      source.indexOf('alt="Grad-CAM overlay"') - 100,
      source.indexOf('alt="Grad-CAM overlay"') + 300,
    );
    expect(overlayImgBlock).toMatch(/opacity:\s*opacity\s*\/\s*100/);
  });
});
