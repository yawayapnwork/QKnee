import { cn } from "@/lib/utils";

/**
 * The one panel primitive in the product. No `backdrop-blur` by default
 * (the old `Card` applied it unconditionally to every instance in the
 * app) -- blur is expensive, communicates nothing, and was never a
 * deliberate choice. Flat surface, one border weight, no unconditional
 * shadow. Use only to group genuinely related information -- not as a
 * default wrapper for every piece of UI.
 */
export function Surface({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("rounded-md border border-surface-3 bg-surface-1", className)}>{children}</div>;
}

export function SurfaceHeader({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("border-b border-surface-3 px-4 py-3", className)}>{children}</div>;
}

export function SurfaceBody({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("px-4 py-3", className)}>{children}</div>;
}
