import { cn } from "@/lib/utils";
import type { ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const variantClasses: Record<Variant, string> = {
  primary:
    "bg-gradient-to-r from-teal-500 to-cyan-400 text-slate-950 font-semibold hover:shadow-glow shadow-md shadow-teal-500/10",
  secondary:
    "bg-slate-800 text-slate-100 ring-1 ring-inset ring-slate-700 hover:bg-slate-700",
  ghost: "text-slate-300 hover:bg-slate-800/60 hover:text-white",
  danger: "bg-rose-500/10 text-rose-400 ring-1 ring-inset ring-rose-500/30 hover:bg-rose-500/20",
};

const sizeClasses: Record<Size, string> = {
  sm: "px-3 py-1.5 text-xs",
  md: "px-4 py-2.5 text-sm",
  lg: "px-6 py-3.5 text-base",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
}

export function Button({ variant = "primary", size = "md", className, ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-50",
        variantClasses[variant],
        sizeClasses[size],
        className,
      )}
      {...props}
    />
  );
}
