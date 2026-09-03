"use client";

import { Download, FileText } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { formatLatency, formatPercent } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

function buildMarkdown(result: DiagnosticResult, caseLabel: string): string {
  const timestamp = new Date().toISOString();
  return `# Q-Knee Automated Screening Report

**Case:** ${caseLabel}
**Generated:** ${timestamp}
**Backend:** ${result.backend} (${result.source})

## Diagnostic Summary

- **Tear Risk Probability:** ${formatPercent(result.riskScore)}
- **Diagnosis:** ${result.diagnosis}
- **Clinical Severity:** ${result.severity}
- **Inference Latency:** ${formatLatency(result.latencyMs)}

## Quantum Circuit Telemetry (⟨Z⟩ expectations)

| Qubit | Expectation |
|-------|-------------|
${result.qubitExpectations.map((v, i) => `| q${i} | ${v.toFixed(4)} |`).join("\n")}

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
    <div className="flex gap-2">
      <Button
        variant="secondary"
        size="sm"
        className="flex-1"
        onClick={() => downloadFile(`qknee-report-${caseLabel}.md`, buildMarkdown(result, caseLabel), "text/markdown")}
      >
        <FileText className="h-3.5 w-3.5" />
        Export Markdown
      </Button>
      <Button
        variant="secondary"
        size="sm"
        className="flex-1"
        onClick={() => window.print()}
      >
        <Download className="h-3.5 w-3.5" />
        Export PDF
      </Button>
    </div>
  );
}
