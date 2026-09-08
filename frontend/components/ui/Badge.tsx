import { cn } from "@/lib/utils";

type BadgeTone = "accent" | "fallback" | "demo" | "proxy" | "neutral" | "live";

const toneClasses: Record<BadgeTone, string> = {
  live: "bg-status-live/10 text-status-live ring-status-live/30",
  demo: "bg-status-demo/10 text-status-demo ring-status-demo/30",
  fallback: "bg-status-fallback/10 text-status-fallback ring-status-fallback/40",
  proxy: "bg-status-proxy/10 text-status-proxy ring-status-proxy/30",
  accent: "bg-accent/10 text-accent ring-accent/30",
  neutral: "bg-surface-2 text-ink-muted ring-surface-3",
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
        "inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-semibold uppercase tracking-wide ring-1 ring-inset",
        toneClasses[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
