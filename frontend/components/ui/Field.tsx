import { forwardRef, useId } from "react";
import type { InputHTMLAttributes, SelectHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

/** A labeled text input with real `htmlFor`/`id` association -- the old
 * `AuthModal` had visually-adjacent `<label>`s with no `for` attribute at
 * all, invisible to screen readers and click-to-focus. */
export const Field = forwardRef<HTMLInputElement, FieldProps>(function Field(
  { label, className, id, ...props },
  ref,
) {
  const generatedId = useId();
  const inputId = id ?? generatedId;
  return (
    <div>
      <label htmlFor={inputId} className="mb-1 block text-xs font-medium text-ink-muted">
        {label}
      </label>
      <input
        ref={ref}
        id={inputId}
        className={cn(
          "w-full rounded-sm border border-surface-3 bg-white px-3 py-2 text-sm text-ink-primary outline-none transition-colors focus:border-accent focus:ring-1 focus:ring-accent",
          className,
        )}
        {...props}
      />
    </div>
  );
});

interface SelectFieldProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label: string;
}

export function SelectField({ label, className, id, children, ...props }: SelectFieldProps) {
  const generatedId = useId();
  const selectId = id ?? generatedId;
  return (
    <div>
      <label htmlFor={selectId} className="mb-1 block text-xs font-medium text-ink-muted">
        {label}
      </label>
      <select
        id={selectId}
        className={cn(
          "w-full rounded-sm border border-surface-3 bg-white px-3 py-2 text-sm text-ink-primary outline-none transition-colors focus:border-accent focus:ring-1 focus:ring-accent",
          className,
        )}
        {...props}
      >
        {children}
      </select>
    </div>
  );
}
