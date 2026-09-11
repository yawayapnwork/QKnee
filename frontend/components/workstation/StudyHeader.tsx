"use client";

import { Wifi, WifiOff, Loader2, PanelLeft } from "lucide-react";
import { ProvenanceBadge } from "@/components/workstation/ProvenanceBadge";
import { ReportExport } from "@/components/workstation/ReportExport";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { Button } from "@/components/ui/Button";
import { Tooltip } from "@/components/ui/Tooltip";
import type { DiagnosticResult } from "@/lib/types";

type ApiHealth = "checking" | "online" | "offline";

/**
 * TOP zone: study identity, live API reachability, the one canonical
 * provenance badge for the current result, and report-export actions —
 * always visible, never competing with the imaging/analysis columns
 * below it. `apiHealth` is fetched once at the page level (see
 * `app/workstation/page.tsx`) and shared with `CaseNav`'s "Live Analysis"
 * label, rather than each component polling `/health` separately.
 * `onOpenCases` is only wired on narrow viewports, where the case list
 * lives in a `Drawer` instead of a permanent column.
 */
export function StudyHeader({
  caseLabel,
  result,
  apiHealth,
  onOpenCases,
}: {
  caseLabel: string;
  result: DiagnosticResult | null;
  apiHealth: ApiHealth;
  onOpenCases: () => void;
}) {
  return (
    <div className="no-print flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-surface-3 bg-surface-1 px-4 py-3 sm:px-6">
      <div className="flex items-center gap-3">
        <Tooltip label="Open case list">
          <Button variant="tertiary" size="sm" className="lg:hidden" onClick={onOpenCases} aria-label="Open case list">
            <PanelLeft className="h-4 w-4" aria-hidden="true" />
          </Button>
        </Tooltip>
        <div>
          <div className="text-2xs uppercase tracking-wide text-ink-faint">Study</div>
          <div className="text-sm font-semibold text-ink-primary">{caseLabel}</div>
        </div>
        <HealthIndicator state={apiHealth} />
      </div>

      <div className="flex items-center gap-3">
        {result && <ProvenanceBadge provenance={result.provenance} compact />}
        {result && <ReportExport result={result} caseLabel={caseLabel} />}
      </div>
    </div>
  );
}

function HealthIndicator({ state }: { state: ApiHealth }) {
  if (state === "checking") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-ink-muted">
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" /> Checking API…
      </span>
    );
  }
  return (
    <StatusIndicator
      tone={state === "online" ? "live" : "demo"}
      icon={state === "online" ? Wifi : WifiOff}
      label={state === "online" ? "API reachable" : "API unreachable"}
      compact
    />
  );
}
