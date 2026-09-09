import { cn } from "@/lib/utils";

// Same colors `--color-severity-*` already alias in `app/globals.css` --
// referenced here as `var(...)` rather than re-picked hex literals, so a
// token change can't silently drift this ring out of sync with every other
// severity-colored element in the app (badges, StatusIndicator, etc).
const TONES: Record<string, string> = {
  Normal: "var(--color-severity-normal)",
  Indeterminate: "var(--color-severity-indeterminate)",
  "Urgent Surgical Consult": "var(--color-severity-urgent)",
};

export function Gauge({ value, severity }: { value: number; severity: string }) {
  const clamped = Math.min(1, Math.max(0, value));
  const circumference = 2 * Math.PI * 54;
  const offset = circumference * (1 - clamped);
  const color = TONES[severity] ?? "var(--color-accent)";
  const percentLabel = `${(clamped * 100).toFixed(1)}%`;

  return (
    <div
      className="flex flex-col items-center gap-2"
      role="img"
      aria-label={`Tear risk ${percentLabel}, severity ${severity}`}
    >
      <div className="relative flex h-32 w-32 shrink-0 items-center justify-center sm:h-36 sm:w-36">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90" aria-hidden="true">
          <circle cx="60" cy="60" r="54" fill="none" stroke="var(--color-surface-3)" strokeWidth="10" />
          <circle
            cx="60"
            cy="60"
            r="54"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-[stroke-dashoffset] duration-700 ease-out"
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="font-mono text-2xl font-bold text-ink-primary sm:text-3xl">{percentLabel}</span>
        </div>
      </div>
      <span
        className={cn("max-w-[10rem] text-center text-[11px] font-semibold uppercase tracking-wide")}
        style={{ color }}
      >
        {severity}
      </span>
    </div>
  );
}
