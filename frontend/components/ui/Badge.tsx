import { cn } from "@/lib/utils";

/**
 * A small inline tag for non-status metadata (e.g. "Hybrid" next to a
 * benchmark row, a role label). Distinct from `StatusIndicator` --
 * `Badge` never renders a LIVE/DEMO/MOCK-FALLBACK provenance state; that
 * is `StatusIndicator`'s/`ProvenanceBadge`'s job exclusively.
 */
type BadgeTone = "accent" | "neutral" | "info" | "warning" | "danger";

const toneClasses: Record<BadgeTone, string> = {
  accent: "bg-accent-subtle text-accent ring-accent/30",
  neutral: "bg-surface-2 text-ink-muted ring-surface-4",
  info: "bg-info/10 text-info ring-info/30",
  warning: "bg-warning/10 text-warning ring-warning/30",
  danger: "bg-danger/10 text-danger ring-danger/30",
};

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-xs px-2 py-0.5 text-2xs font-semibold uppercase tracking-wide ring-1 ring-inset",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
