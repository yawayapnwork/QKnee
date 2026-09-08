import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes } from "react";

/**
 * The four-step action hierarchy (DESIGN_SYSTEM.md §7):
 *   primary     — the one action a screen wants you to take (flat accent
 *                 fill; never more than one `primary` button visible at once)
 *   secondary   — an alternative, equally-valid action (outlined)
 *   tertiary    — a low-emphasis/auxiliary action (text-only, no border)
 *   destructive — an irreversible or credential-affecting action (this
 *                 product currently has none exposed to the UI, but the
 *                 variant exists so one is never improvised from `danger`-
 *                 tinted `secondary` styling)
 * No gradient, no glow on any variant -- a flat accent fill is the entire
 * "primary" signal.
 */
type Variant = "primary" | "secondary" | "tertiary" | "destructive";
type Size = "sm" | "md";

const variantClasses: Record<Variant, string> = {
  primary: "bg-accent text-surface-0 font-semibold hover:bg-accent-strong",
  secondary: "bg-transparent text-ink-primary ring-1 ring-inset ring-surface-4 hover:bg-surface-2",
  tertiary: "bg-transparent text-ink-muted hover:bg-surface-2 hover:text-ink-primary",
  destructive: "bg-transparent text-danger ring-1 ring-inset ring-danger/40 hover:bg-danger/10",
};

const sizeClasses: Record<Size, string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2 text-sm",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export function Button({ variant = "primary", size = "md", className, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-sm transition-colors duration-fast disabled:cursor-not-allowed disabled:opacity-50",
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
      {...props}
    />
  );
}
