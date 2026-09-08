import { Info, AlertTriangle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * The one inline-message primitive: INFORMATION / WARNING / ERROR
 * (DESIGN_SYSTEM.md §11). Replaces the ad-hoc `<div className="rounded-lg
 * border border-amber-500/30 ...">`-style banners that used to be
 * hand-copied into `AuthModal`, `StudySelector`, and the workstation page
 * separately, each with its own slightly different color/spacing.
 */
type Tone = "info" | "warning" | "error";

const TONE_STYLES: Record<Tone, { className: string; icon: typeof Info }> = {
  info: { className: "border-info/30 bg-info/10 text-info", icon: Info },
  warning: { className: "border-warning/30 bg-warning/10 text-warning", icon: AlertTriangle },
  error: { className: "border-danger/30 bg-danger/10 text-danger", icon: XCircle },
};

export function Alert({ tone, children, className }: { tone: Tone; children: React.ReactNode; className?: string }) {
  const { className: toneClassName, icon: Icon } = TONE_STYLES[tone];
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={cn("flex items-start gap-2 rounded-sm border px-3 py-2 text-xs", toneClassName, className)}
    >
      <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      <div>{children}</div>
    </div>
  );
}
