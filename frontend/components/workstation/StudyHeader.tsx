"use client";

import { useEffect, useState } from "react";
import { Wifi, WifiOff, Loader2, Upload } from "lucide-react";
import { fetchHealth } from "@/lib/api";
import { ProvenanceBadge } from "@/components/workstation/ProvenanceBadge";
import type { DiagnosticResult } from "@/lib/types";

type HealthState = "checking" | "online" | "offline";

/**
 * TOP zone (FRONTEND_REDESIGN.md Part 4, Workstation wireframe): study
 * identity, live API reachability, the one canonical provenance badge for
 * the current result, and the upload action -- always visible, never
 * competing with the imaging/analysis columns below it.
 */
export function StudyHeader({
  caseLabel,
  result,
  onUpload,
}: {
  caseLabel: string;
  result: DiagnosticResult | null;
  onUpload: (file: File) => void;
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
      <div className="flex items-center gap-4">
        <div>
          <div className="text-xs uppercase tracking-wide text-ink-faint">Study</div>
          <div className="text-sm font-semibold text-ink-primary">{caseLabel}</div>
        </div>
        <HealthIndicator state={health} />
      </div>

      <div className="flex items-center gap-3">
        {result && <ProvenanceBadge provenance={result.provenance} compact />}
        <label className="flex cursor-pointer items-center gap-1.5 rounded-md border border-surface-3 px-3 py-1.5 text-xs font-medium text-ink-primary hover:bg-surface-2">
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
  if (state === "online") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-status-live">
        <Wifi className="h-3.5 w-3.5" aria-hidden="true" /> API reachable
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs text-status-demo">
      <WifiOff className="h-3.5 w-3.5" aria-hidden="true" /> API unreachable
    </span>
  );
}
