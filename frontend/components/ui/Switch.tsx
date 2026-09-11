import { cn } from "@/lib/utils";

/** A real toggle switch -- `role="switch"` + `aria-checked`, not a
 * `<button>` styled to look like one with no state exposed to assistive
 * tech. */
export function Switch({
  checked,
  onChange,
  disabled,
  label,
}: {
  checked: boolean;
  onChange: () => void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={onChange}
      className={cn(
        "relative h-5 w-9 rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-40",
        checked && !disabled ? "bg-accent" : "bg-surface-3",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white transition-transform",
          checked && !disabled ? "translate-x-4" : "translate-x-0",
        )}
      />
    </button>
  );
}
