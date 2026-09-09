import { cn } from "@/lib/utils";

type Size = "xs" | "sm" | "md";

/**
 * The one "numbered step" chip used everywhere a sequence needs a small
 * digit badge -- `QuantumTelemetry`'s circuit-architecture stages, and
 * `ExplanationWorkspace`'s section headers and TOC nav. Previously each
 * site hand-rolled its own box (three different sizes, and the TOC one
 * used an arbitrary `rounded-[3px]` instead of the design system's
 * `--radius-xs` token it's numerically identical to). One component, three
 * deliberate size variants -- not one arbitrary value per call site.
 */
const SIZE_CLASSES: Record<Size, string> = {
  xs: "h-4 w-4 rounded-xs text-[9px]",
  sm: "h-5 w-5 rounded-sm text-[10px]",
  md: "h-6 w-6 rounded-sm text-xs",
};

export function PipelineStepBadge({
  index,
  size = "sm",
  className,
}: {
  index: number | string;
  size?: Size;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "flex shrink-0 items-center justify-center border border-surface-4 font-mono text-ink-muted",
        SIZE_CLASSES[size],
        className,
      )}
    >
      {index}
    </span>
  );
}
