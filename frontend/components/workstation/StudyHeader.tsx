"use client";

import { useEffect, useState } from "react";
import { Wifi, WifiOff, Loader2, Upload, PanelLeft } from "lucide-react";
import { fetchHealth } from "@/lib/api";
import { ProvenanceBadge } from "@/components/workstation/ProvenanceBadge";
import { StatusIndicator } from "@/components/ui/StatusIndicator";
import { Button } from "@/components/ui/Button";
import { Tooltip } from "@/components/ui/Tooltip";
import type { DiagnosticResult } from "@/lib/types";

type HealthState = "checking" | "online" | "offline";

/**
 * TOP zone: study identity, live API reachability, the one canonical
 * provenance badge for the current result, and the upload action --
 * always visible, never competing with the imaging/analysis columns
 * below it. `onOpenCases` is only wired on narrow viewports, where the
 * case list lives in a `Drawer` instead of a permanent column.
 */
export function StudyHeader({
  caseLabel,
  result,
  onUpload,
  onOpenCases,
}: {
  caseLabel: string;
  result: DiagnosticResult | null;
  onUpload: (file: File) => void;
  onOpenCases: () => void;
}) {
  const [health, setHealth] = useState<HealthState>("checking");

  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    fetchHealth(controller.signal)
      .then(() => setHealth("online"))
      .catch(() => setHealth("offline"))
      .finally(() => clearTimeout(timeout));
    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-surface-3 bg-surface-1 px-4 py-3 sm:px-6">
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
        <HealthIndicator state={health} />
      </div>

      <div className="flex items-center gap-3">
        {result && <ProvenanceBadge provenance={result.provenance} compact />}
        <label className="flex cursor-pointer items-center gap-1.5 rounded-sm border border-surface-3 px-3 py-1.5 text-xs font-medium text-ink-primary hover:bg-surface-2">
          <Upload className="h-3.5 w-3.5" aria-hidden="true" />
          Upload .dcm / .npy
          <input
            type="file"
            accept=".dcm,.dicom,.npy"
            className="sr-only"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onUpload(file);
            }}
          />
        </label>
      </div>
    </div>
  );
}

function HealthIndicator({ state }: { state: HealthState }) {
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
