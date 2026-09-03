import { Loader2, AlertTriangle } from "lucide-react";
import { Card, CardHeader, CardBody } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Gauge } from "@/components/ui/Gauge";
import { QuantumTelemetry } from "@/components/workstation/QuantumTelemetry";
import { ReportExport } from "@/components/workstation/ReportExport";
import { formatLatency } from "@/lib/utils";
import type { DiagnosticResult } from "@/lib/types";

const SEVERITY_TONE = {
  Normal: "emerald",
  Indeterminate: "amber",
  "Urgent Surgical Consult": "rose",
} as const;

export function TriageCard({
  result,
  loading,
  caseLabel,
  canDiagnose,
}: {
  result: DiagnosticResult | null;
  loading: boolean;
  caseLabel: string;
  canDiagnose: boolean;
}) {
  return (
    <Card className="flex flex-col">
      <CardHeader className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-100">Diagnostic Triage Card</h3>
        {result && <Badge tone={SEVERITY_TONE[result.severity]}>{result.severity}</Badge>}
      </CardHeader>

      <CardBody className="flex flex-1 flex-col gap-6">
        {!canDiagnose && (
          <div className="flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-400">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            Signed in as a Research Observer — inference results shown are read-only reference data.
            Radiologist credentials are required to run new predictions against the live backend.
          </div>
        )}

        {loading ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-3 py-10 text-slate-500">
            <Loader2 className="h-6 w-6 animate-spin text-teal-400" />
            <span className="text-xs">Running hybrid inference…</span>
          </div>
        ) : result ? (
          <>
            <div className="flex justify-center">
              <Gauge value={result.riskScore} severity={result.severity} />
            </div>

            <div className="grid grid-cols-3 gap-3 text-center">
              <Metric label="Diagnosis" value={result.diagnosis} />
              <Metric label="Latency" value={formatLatency(result.latencyMs)} />
              <Metric label="Backend" value={result.source === "live" ? "Live" : "Preset"} />
            </div>

            <QuantumTelemetry expectations={result.qubitExpectations} />

            <div className="mt-auto pt-2">
              <ReportExport result={result} caseLabel={caseLabel} />
            </div>
          </>
        ) : (
          <div className="flex flex-1 items-center justify-center py-10 text-xs text-slate-600">
            No diagnostic result yet — select a sample case or upload a scan.
          </div>
        )}
      </CardBody>
    </Card>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 px-2 py-2.5">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-0.5 truncate font-mono text-xs font-semibold text-slate-200">{value}</div>
    </div>
  );
}
