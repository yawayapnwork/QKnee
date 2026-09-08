import { Alert } from "@/components/ui/Alert";
import { PRESET_CASES } from "@/lib/mock-data";
import type { PresetCase } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * LEFT zone (permanent column at `lg:` and above; rendered inside a
 * `Drawer` below it). A real radio group -- every item is a focusable,
 * labeled button, not a hand-rolled `<div>` dropdown with no keyboard
 * support.
 */
export function StudySelector({
  activeCaseId,
  onSelectCase,
  canDiagnose,
}: {
  activeCaseId: string;
  onSelectCase: (preset: PresetCase) => void;
  canDiagnose: boolean;
}) {
  return (
    <div className="flex flex-col gap-3 p-4">
      <div>
        <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Demo Cases</h2>
        <p className="mt-1 text-xs text-ink-muted">Precomputed reference studies, clearly marked as demo data.</p>
      </div>

      <div role="radiogroup" aria-label="Demo case" className="flex flex-col gap-1.5">
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
              <div className="font-medium">{preset.label}</div>
              <div className="text-ink-faint">{preset.description}</div>
            </button>
          );
        })}
      </div>

      {!canDiagnose && (
        <Alert tone="info" className="mt-2">
          Sign in with radiologist credentials to run a new upload against the live model. Demo cases remain
          available to everyone.
        </Alert>
      )}
    </div>
  );
}
