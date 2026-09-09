import { Upload } from "lucide-react";
import { Alert } from "@/components/ui/Alert";
import { Badge } from "@/components/ui/Badge";
import { PRESET_CASES } from "@/lib/mock-data";
import type { PresetCase } from "@/lib/types";
import { cn } from "@/lib/utils";

type ApiHealth = "checking" | "online" | "offline";

/**
 * LEFT zone: case/study navigation, replacing `StudySelector.tsx`. Two
 * sections, not one — "Demo Cases" (every row tagged `PRECOMPUTED DEMO`,
 * never presented as if it were a live result) and "Upload Study" (the
 * live-analysis entry point, consolidated here instead of living
 * separately in the header, so there is exactly one place a viewer looks
 * for "how do I get a result").
 *
 * The upload control's label only ever says "LIVE ANALYSIS" when BOTH a
 * real model is authorized for this viewer (`canDiagnose`, a radiologist
 * session) AND the API has actually confirmed reachable (`apiHealth ===
 * "online"`) — while health is still `"checking"` or has come back
 * `"offline"`, the control reads the honest, unclaimed "Upload Study"
 * instead. The upload input itself stays clickable either way; a failed
 * live request is handled by `ErrorState`, never a silent downgrade here.
 */
export function CaseNav({
  activeCaseId,
  onSelectCase,
  canDiagnose,
  apiHealth,
  onUpload,
}: {
  activeCaseId: string;
  onSelectCase: (preset: PresetCase) => void;
  canDiagnose: boolean;
  apiHealth: ApiHealth;
  onUpload: (file: File) => void;
}) {
  const liveAnalysisAvailable = canDiagnose && apiHealth === "online";

  return (
    <div className="flex flex-col gap-6 p-4">
      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Demo Cases</h2>
        <p className="mt-1 text-xs text-ink-muted">Precomputed reference studies, clearly marked as demo data.</p>

        <div role="radiogroup" aria-label="Demo case" className="mt-3 flex flex-col gap-1.5">
          {PRESET_CASES.map((preset) => {
            const isActive = preset.id === activeCaseId;
            return (
              <button
                key={preset.id}
                type="button"
                role="radio"
                aria-checked={isActive}
                onClick={() => onSelectCase(preset)}
                className={cn(
                  "rounded-md border px-3 py-2 text-left text-xs transition-colors",
                  isActive
                    ? "border-accent bg-accent/10 text-ink-primary"
                    : "border-surface-3 text-ink-muted hover:border-surface-3 hover:bg-surface-2",
                )}
              >
                <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
                  <span className="font-medium">{preset.label}</span>
                  <Badge tone="neutral" className="shrink-0">
                    Precomputed Demo
                  </Badge>
                </div>
                <div className="mt-0.5 text-ink-faint">{preset.description}</div>
              </button>
            );
          })}
        </div>
      </section>

      <section className="border-t border-surface-3 pt-4">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Upload Study</h2>
        <p className="mt-1 text-xs text-ink-muted">
          {!canDiagnose
            ? "Sign in with radiologist credentials to run a live upload against the model."
            : apiHealth === "offline"
              ? "The API is currently unreachable — live analysis is not available right now."
              : "Runs a live model inference against the uploaded volume."}
        </p>

        <label
          className={cn(
            "mt-3 flex cursor-pointer items-center justify-center gap-2 rounded-md border border-dashed border-surface-4 px-3 py-3 text-center text-xs font-medium text-ink-primary transition-colors hover:bg-surface-2",
          )}
        >
          <Upload className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          {liveAnalysisAvailable ? "Live Analysis — Upload .dcm / .npy" : "Upload .dcm / .npy"}
          <input
            type="file"
            accept=".dcm,.dicom,.npy"
            className="sr-only"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onUpload(file);
              e.target.value = "";
            }}
          />
        </label>

        {!canDiagnose && (
          <Alert tone="info" className="mt-2">
            Demo cases remain available to everyone without signing in.
          </Alert>
        )}
      </section>
    </div>
  );
}
