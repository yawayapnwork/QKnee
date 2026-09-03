import { cn } from "@/lib/utils";

type BadgeTone = "teal" | "rose" | "emerald" | "amber" | "slate";

const toneClasses: Record<BadgeTone, string> = {
  teal: "bg-teal-500/10 text-teal-400 ring-teal-500/30",
  rose: "bg-rose-500/10 text-rose-400 ring-rose-500/30",
  emerald: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30",
  amber: "bg-amber-500/10 text-amber-400 ring-amber-500/30",
  slate: "bg-slate-500/10 text-slate-300 ring-slate-500/30",
};

export function Badge({
  children,
  tone = "slate",
  className,
  dot,
}: {
  children: React.ReactNode;
  tone?: BadgeTone;
  className?: string;
  dot?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset",
        toneClasses[tone],
        className,
      )}
    >
      {dot && <span className={cn("h-1.5 w-1.5 rounded-full", `bg-current`)} />}
      {children}
    </span>
  );
}
