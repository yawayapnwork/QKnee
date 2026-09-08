import { cn } from "@/lib/utils";

export interface TabItem<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
  disabledReason?: string;
}

/** A real ARIA tablist -- the plane selector used to be a row of plain
 * `<button>`s with no group semantics at all. Each tab announces its own
 * selected/disabled state; a disabled tab's reason is always visible
 * (via `title` AND inline, never hover-only), never just implied by
 * being greyed out. */
export function Tabs<T extends string>({
  items,
  value,
  onChange,
  label,
}: {
  items: TabItem<T>[];
  value: T;
  onChange: (next: T) => void;
  label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex gap-1 rounded-md bg-surface-0 p-1">
      {items.map((item) => {
        const isActive = item.value === value;
        return (
          <button
            key={item.value}
            role="tab"
            type="button"
            aria-selected={isActive}
            aria-disabled={item.disabled}
            disabled={item.disabled}
            title={item.disabled ? item.disabledReason : undefined}
            onClick={() => !item.disabled && onChange(item.value)}
            className={cn(
              "rounded px-3 py-1.5 text-xs font-medium transition-colors",
              item.disabled && "cursor-not-allowed text-ink-faint opacity-50",
              !item.disabled && isActive && "bg-accent text-surface-0",
              !item.disabled && !isActive && "text-ink-muted hover:text-ink-primary",
            )}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
