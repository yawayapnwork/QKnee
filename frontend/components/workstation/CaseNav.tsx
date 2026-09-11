import { Loader2, Upload } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import type { CaseSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

type ApiHealth = "checking" | "online" | "offline";

/**
 * LEFT zone: case/study navigation. Two sections:
 * "Demo Cases" (real RSNA Knee studies, scored once offline)
 * and "Upload Study" (the live-analysis entry point).
 *
 * The upload control reads "Live Analysis — Upload .dcm / .npy" when
 * the API has confirmed reachable (`apiHealth === "online"`), and
 * "Upload .dcm / .npy" otherwise.
 */
export function CaseNav({
  cases,
  activeCaseId,
  onSelectCase,
  canDiagnose = true,
  apiHealth,
  onUpload,
  isUploading = false,
}: {
  cases: CaseSummary[];
  activeCaseId: string | null;
  onSelectCase: (caseId: string) => void;
  canDiagnose?: boolean;
  apiHealth: ApiHealth;
  onUpload: (file: File) => void;
  isUploading?: boolean;
}) {
  const liveAnalysisAvailable = apiHealth === "online";

  return (
    <div className="flex flex-col gap-6 p-4">
      <section>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Demo Cases</h2>
        <p className="mt-1 text-xs text-ink-muted">
          Real RSNA Knee studies, scored once offline — clearly marked as demo data.
        </p>

        {cases.length === 0 ? (
          <p className="mt-3 text-xs text-ink-faint">No demo cases available right now.</p>
        ) : (
          <div role="radiogroup" aria-label="Demo case" className="mt-3 flex flex-col gap-1.5">
            {cases.map((c) => {
              const isActive = c.case_id === activeCaseId;
              return (
                <button
                  key={c.case_id}
                  type="button"
                  role="radio"
                  aria-checked={isActive}
                  onClick={() => onSelectCase(c.case_id)}
                  className={cn(
                    "rounded-md border px-3 py-2 text-left text-xs transition-colors",
                    isActive
                      ? "border-accent bg-accent-subtle font-medium text-ink-primary shadow-xs ring-1 ring-accent/30"
                      : "border-surface-3 bg-white text-ink-secondary hover:border-surface-4 hover:bg-surface-2 hover:text-ink-primary",
                  )}
                >
                  <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1">
                    <span className="font-medium">{c.label}</span>
                    <Badge tone="neutral" className="shrink-0">
                      Precomputed Demo
                    </Badge>
                  </div>
                  <div className="mt-0.5 text-ink-faint">{c.diagnosis}</div>
                </button>
              );
            })}
          </div>
        )}
      </section>

      <section className="border-t border-surface-3 pt-4">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Upload Study</h2>
        <p className="mt-1 text-xs text-ink-muted">
          {apiHealth === "offline"
            ? "The API is currently unreachable — live analysis is not available right now."
            : "Runs a live model inference against the uploaded volume."}
        </p>

        <label
          className={cn(
            "relative mt-3 flex items-center justify-center gap-2 rounded-md border border-dashed border-surface-3 bg-surface-2/40 px-3 py-3 text-center text-xs font-medium text-ink-primary transition-colors",
            isUploading
              ? "cursor-wait opacity-70"
              : "cursor-pointer hover:border-accent hover:bg-surface-2 hover:text-accent-strong",
          )}
        >
          {isUploading ? (
            <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-accent" aria-hidden="true" />
          ) : (
            <Upload className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
          )}
          {isUploading
            ? "Analyzing scan volume..."
            : liveAnalysisAvailable
              ? "Live Analysis — Upload .dcm / .npy"
              : "Upload .dcm / .npy"}
          <input
            type="file"
            accept=".dcm,.dicom,.npy"
            disabled={isUploading}
            className="sr-only"
            onChange={(e) => {
              e.stopPropagation();
              const file = e.target.files?.[0];
              if (file) onUpload(file);
              e.target.value = "";
            }}
          />
        </label>
      </section>
    </div>
  );
}
