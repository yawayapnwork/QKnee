import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

// No gradient, no glow. A flat accent fill is the entire "primary" signal --
// it does not need a shadow effect to read as the main action.
const variantClasses: Record<Variant, string> = {
  primary: "bg-accent text-surface-0 font-semibold hover:bg-accent-muted hover:text-ink-primary",
  secondary: "bg-surface-2 text-ink-primary ring-1 ring-inset ring-surface-3 hover:bg-surface-3",
  ghost: "text-ink-muted hover:bg-surface-2 hover:text-ink-primary",
  danger: "bg-status-fallback/10 text-status-fallback ring-1 ring-inset ring-status-fallback/40 hover:bg-status-fallback/20",
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
        "inline-flex items-center justify-center gap-2 rounded-md transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-50",
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
      {...props}
    />
  );
}
