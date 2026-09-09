"use client";

import { useEffect, useState } from "react";
import { ProvenanceBadge } from "@/components/workstation/ProvenanceBadge";
import { GRADCAM_DISCLAIMER, LIMITATIONS } from "@/lib/explanation-copy";
import { PLANE_LABELS, sliceImageSrc, toImageSrc } from "@/lib/viewer";
import { formatLatency, formatPercent } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

/**
 * The ONLY thing `window.print()` should ever actually print. Every other
 * on-screen element in the workstation carries `.no-print` (see
 * `app/globals.css`); this component is the sole `.print-only` block, and
 * it is deliberately built from plain text and two flat `<img>`s rather
 * than the interactive viewer/panels -- no zoom state, no slider, no
 * button, nothing that only makes sense on a screen a mouse can reach.
 *
 * Every field is either a real value read off `result`/`caseLabel`, or an
 * explicit "not available" -- this file invents no clinical metadata (no
 * fabricated run ID, no invented facility name, nothing). "Generated"
 * below is the one exception worth calling out: it's a real fact about
 * when the report was produced, not a claim about the study itself.
 *
 * That timestamp is deliberately NOT `new Date().toISOString()` evaluated
 * directly in the render body -- this component participates in Next.js's
 * normal server-render-then-hydrate cycle (it's a plain import from the
 * `"use client"` workstation page, not behind `dynamic(..., {ssr:false})`),
 * so a value computed at render time would differ between the server's
 * render pass and the client's hydration pass a moment later -- two
 * different clock reads, two different strings, one guaranteed hydration
 * mismatch (confirmed live: this was exactly what threw "Text content
 * does not match server-rendered HTML" in the browser console before this
 * fix). Reading it in `useEffect` instead means the server and the
 * client's FIRST render both produce the identical "—" placeholder --
 * hydration diffs against that, succeeds, and only then does a normal
 * post-hydration state update swap in the real timestamp. Printing is a
 * client-only action (`window.print()`) triggered well after mount, so
 * there is no real cost to the one-tick delay.
 */
export function PrintReport({ result, caseLabel }: { result: DiagnosticResult; caseLabel: string }) {
  const [timestamp, setTimestamp] = useState<string | null>(null);
  useEffect(() => {
    setTimestamp(new Date().toISOString());
  }, []);
  const volume = result.volume;
  const baseImage = sliceImageSrc(volume, volume.primaryPlane, volume.primarySliceIndex);
  const hasGradcam = Boolean(volume.gradcamOverlay && volume.gradcamPlane !== null);
  const gradcamImage = hasGradcam ? toImageSrc(volume.gradcamOverlay as string) : null;

  return (
    <div className="print-only px-8 py-8 text-black">
      <h1 className="text-xl font-bold">Q-Knee Automated Screening Report</h1>
      <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
        <dt className="font-semibold">Case</dt>
        <dd>{caseLabel}</dd>
        <dt className="font-semibold">Generated</dt>
        <dd>{timestamp ?? "—"}</dd>
      </dl>

      <div className="mt-4">
        <ProvenanceBadge provenance={result.provenance} />
      </div>

      <h2 className="mt-6 text-sm font-semibold uppercase tracking-wide">Prediction</h2>
      <dl className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
        <dt>Risk score</dt>
        <dd>{formatPercent(result.riskScore)}</dd>
        <dt>Diagnosis</dt>
        <dd>{result.diagnosis}</dd>
        <dt>Severity</dt>
        <dd>{result.severity}</dd>
        <dt>Inference latency</dt>
        <dd>{formatLatency(result.latencyMs)}</dd>
        <dt>Model status</dt>
        <dd>{result.provenance.modelSourceLabel ?? "Not reported"}</dd>
      </dl>

      <h2 className="mt-6 text-sm font-semibold uppercase tracking-wide">MRI slice</h2>
      {baseImage ? (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element -- base64 data URI */}
          <img src={baseImage} alt="MRI slice" className="mt-2 h-64 w-64 border border-black/40 object-contain" />
          <p className="mt-1 text-2xs">
            Plane: {PLANE_LABELS[volume.primaryPlane]} · Slice: {volume.primarySliceIndex + 1} /{" "}
            {volume.planes[volume.primaryPlane].numSlices}
          </p>
        </>
      ) : (
        <p className="mt-2 text-xs">No MRI slice available for this result.</p>
      )}

      <h2 className="mt-6 text-sm font-semibold uppercase tracking-wide">Grad-CAM attention</h2>
      {gradcamImage ? (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element -- base64 data URI */}
          <img
            src={gradcamImage}
            alt="Grad-CAM model attention visualization"
            className="mt-2 h-64 w-64 border border-black/40 object-contain"
          />
          <p className="mt-2 max-w-lg text-2xs leading-relaxed">{GRADCAM_DISCLAIMER}</p>
        </>
      ) : (
        <p className="mt-2 text-xs">No Grad-CAM overlay was computed for this result.</p>
      )}

      <h2 className="mt-6 text-sm font-semibold uppercase tracking-wide">Limitations</h2>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-xs">
        {LIMITATIONS.map((item) => (
          <li key={item.title}>
            <span className="font-semibold">{item.title}.</span> {item.detail}
          </li>
        ))}
      </ul>

      <p className="mt-6 max-w-lg text-2xs leading-relaxed">
        Investigational research prototype output — not for clinical use. Findings require independent review by a
        licensed radiologist or orthopedic clinician.
      </p>
    </div>
  );
}
