import type { DiagnosticResult, PresetCase, SeverityTag } from "./types";

export const PRESET_CASES: PresetCase[] = [
  {
    id: "sample-01",
    label: "Sample 01",
    description: "Confirmed ACL Tear",
    category: "ACL Tear",
    riskScore: 0.912,
    qubitExpectations: [-0.71, 0.62, -0.48, 0.83],
  },
  {
    id: "sample-02",
    label: "Sample 02",
    description: "Intact Meniscus",
    category: "Intact Meniscus",
    riskScore: 0.081,
    qubitExpectations: [0.12, -0.05, 0.09, -0.14],
  },
  {
    id: "sample-03",
    label: "Sample 03",
    description: "Complex Multi-Compartment Defect",
    category: "Multi-Compartment Defect",
    riskScore: 0.584,
    qubitExpectations: [-0.01, 0.19, -0.05, -0.08],
  },
];

export function severityFromRisk(riskScore: number): SeverityTag {
  if (riskScore >= 0.75) return "Urgent Surgical Consult";
  if (riskScore >= 0.4) return "Indeterminate";
  return "Normal";
}

/** Deterministic 1x1-pixel-scaled placeholder heatmap (soft radial gradient encoded as SVG data URI)
 * used when the live backend is cold-starting on Render's free tier and a visual placeholder is
 * needed instead of a blank panel. */
export function placeholderHeatmapDataUri(seedHex: string): string {
  const hue = parseInt(seedHex.slice(0, 2), 16) % 360;
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='320' height='320'>
    <defs>
      <radialGradient id='g' cx='50%' cy='50%' r='60%'>
        <stop offset='0%' stop-color='hsl(${hue},85%,55%)' stop-opacity='0.85'/>
        <stop offset='60%' stop-color='hsl(${hue + 20},80%,45%)' stop-opacity='0.35'/>
        <stop offset='100%' stop-color='#0f172a' stop-opacity='0'/>
      </radialGradient>
    </defs>
    <rect width='320' height='320' fill='#0f172a'/>
    <circle cx='160' cy='160' r='150' fill='url(#g)'/>
  </svg>`;
  return `data:image/svg+xml;base64,${btoa(svg)}`;
}

export function mockDiagnosticResult(preset: PresetCase): DiagnosticResult {
  const seed = preset.id.replace(/\D/g, "").padStart(2, "0");
  return {
    riskScore: preset.riskScore,
    diagnosis: preset.riskScore >= 0.5 ? "Tear Detected" : "Normal",
    severity: severityFromRisk(preset.riskScore),
    heatmap: placeholderHeatmapDataUri(seed),
    backend: "mock/cold-start-fallback",
    latencyMs: null,
    qubitExpectations: preset.qubitExpectations,
    source: "mock",
  };
}
