"use client";

import { Download, Printer } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { hasQuantumTelemetry } from "@/lib/quantum-telemetry";
import { formatLatency, formatPercent } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

/**
 * Builds the exported report from `result.provenance` -- the same
 * canonical model the on-screen badge reads from. This file used to
 * maintain its OWN, third, hardcoded copy of the provenance label map and
 * cite the legacy `backend`/`source` fields instead; a viewer could see
 * "LIVE" on screen and a differently-worded label in the downloaded
 * report for the identical result. That divergence is now structurally
 * impossible: there is only one label to read.
 */
function buildMarkdown(result: DiagnosticResult, caseLabel: string): string {
  const timestamp = new Date().toISOString();
  const telemetry = result.quantumTelemetry;
  // Same "does data exist" question `QuantumTelemetry.tsx` asks -- a
  // precomputed demo case's real, stored per-qubit values are still
  // exported (never hidden), just under the honest "NOT INDEPENDENTLY
  // VERIFIABLE" label rather than "QUANTUM SIMULATOR".
  const telemetrySection = hasQuantumTelemetry(telemetry)
    ? `| Qubit | Expectation |\n|-------|-------------|\n${telemetry.expectations
        .map((v, i) => `| q${i} | ${v.toFixed(4)} |`)
        .join("\n")}`
    : `_${result.provenance.quantumExecutionLabel}._`;

  return `# Q-Knee Automated Screening Report

**Case:** ${caseLabel}
**Generated:** ${timestamp}
**Provenance:** ${result.provenance.provenanceLabel}
**Model:** ${result.provenance.modelSourceLabel ?? "n/a"}
**Quantum execution:** ${result.provenance.quantumExecutionLabel}

## Diagnostic Summary

- **Tear Risk Probability:** ${formatPercent(result.riskScore)}
- **Diagnosis:** ${result.diagnosis}
- **Clinical Severity:** ${result.severity}
- **Inference Latency:** ${formatLatency(result.latencyMs)}

## Quantum Circuit Telemetry (⟨Z⟩ expectations)

${telemetrySection}

---

*Investigational research prototype output — not for clinical use. Findings require independent
review by a licensed radiologist or orthopedic clinician.*
`;
}

function downloadFile(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function ReportExport({ result, caseLabel }: { result: DiagnosticResult; caseLabel: string }) {
  return (
    <div className="no-print flex gap-2">
      <Button
        variant="secondary"
        size="sm"
        className="flex-1"
        onClick={() => downloadFile(`qknee-report-${caseLabel}.md`, buildMarkdown(result, caseLabel), "text/markdown")}
      >
        <Download className="h-3.5 w-3.5" aria-hidden="true" />
        Export Markdown
      </Button>
      {/* Honestly labeled: this opens the browser print dialog, which lets
          the user save as PDF -- it does not generate a PDF file itself.
          The old label ("Export PDF") implied server-side/library PDF
          generation that never existed. */}
      <Button variant="secondary" size="sm" className="flex-1" onClick={() => window.print()}>
        <Printer className="h-3.5 w-3.5" aria-hidden="true" />
        Print / Save as PDF
      </Button>
    </div>
  );
}
