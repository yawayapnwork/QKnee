import { useId } from "react";
import { cn } from "@/lib/utils";

/**
 * A minimal, dependency-free tooltip: pure CSS group-hover/focus-within,
 * no positioning library. Used for disabled-control reasons (e.g. an
 * unavailable imaging plane) so the reason is discoverable by mouse AND
 * keyboard -- never hover-only, and never the sole place a critical
 * reason is stated (the disabled plane tab's reason is also in its own
 * `aria-label`/visible microcopy where space allows).
 */
export function Tooltip({ label, children }: { label: string; children: React.ReactNode }) {
  const id = useId();
  return (
    <span className="group relative inline-flex">
      <span aria-describedby={id}>{children}</span>
      <span
        role="tooltip"
        id={id}
        className={cn(
          "pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 w-max max-w-[220px] -translate-x-1/2 rounded-sm border border-surface-3 bg-white px-2.5 py-1 text-2xs text-ink-primary opacity-0 shadow-2 transition-opacity duration-fast",
          "group-hover:opacity-100 group-focus-within:opacity-100",
        )}
      >
        {label}
      </span>
    </span>
  );
}
