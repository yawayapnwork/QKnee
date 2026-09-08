import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

/**
 * The ONE primitive allowed to render a colored status dot/label anywhere
 * in the product -- API reachability, model checkpoint availability, and
 * (via `ProvenanceBadge`, which wraps this) prediction provenance all
 * flow through it. Five tones, exhaustively typed, each with a fixed
 * icon+color+shadow combination a caller cannot partially override --
 * this is the structural guarantee that MOCK can never accidentally
 * render with LIVE's styling: there is no "pass any hex you want" prop,
 * only a closed set of named tones.
 */
export type StatusTone = "live" | "demo" | "fallback" | "proxy" | "neutral";

const TONE_STYLES: Record<StatusTone, { className: string }> = {
  live: { className: "border-status-live/40 bg-status-live/10 text-status-live" },
  demo: { className: "border-status-demo/40 bg-status-demo/10 text-status-demo" },
  // The loudest, highest-contrast tone on purpose -- the only one paired
  // with a persistent shadow ring, so it is never confusable with any
  // other status even at a glance or in a screenshot thumbnail.
  fallback: { className: "border-danger bg-danger/15 text-danger shadow-alert" },
  proxy: { className: "border-status-proxy/40 bg-status-proxy/10 text-status-proxy" },
  neutral: { className: "border-surface-4 bg-surface-2 text-ink-muted" },
};

export function StatusIndicator({
  tone,
  label,
  icon: Icon,
  detail,
  compact = false,
}: {
  tone: StatusTone;
  label: string;
  icon: LucideIcon;
  detail?: string;
  compact?: boolean;
}) {
  const { className } = TONE_STYLES[tone];

  const badge = (
    <span
      role="status"
      className={cn(
        "inline-flex w-fit items-center gap-1.5 rounded-sm border px-2.5 py-1 text-xs font-bold tracking-wide",
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </span>
  );

  if (compact || !detail) return badge;

  return (
    <div className="flex flex-col gap-1">
      {badge}
      <span className="text-2xs text-ink-faint">{detail}</span>
    </div>
  );
}
