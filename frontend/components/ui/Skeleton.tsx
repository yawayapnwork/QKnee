import { cn } from "@/lib/utils";

/** A shape-matched loading placeholder -- used instead of a centered
 * spinner so the layout doesn't jump when data arrives. A single
 * `animate-pulse` (Tailwind's built-in, tied to actual pending state, not
 * a permanent decorative loop) is the only motion. */
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-surface-2", className)} aria-hidden="true" />;
}
