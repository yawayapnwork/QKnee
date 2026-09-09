import { useRef } from "react";
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
 * being greyed out. Arrow-key roving focus follows the WAI-ARIA tablist
 * pattern (Left/Right move focus, wrapping at the ends, skipping disabled
 * tabs, Home/End jump to the first/last enabled tab) -- Tab order still
 * reaches the group in one stop, matching native tab widgets rather than
 * requiring one Tab press per plane. */
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
  const buttonRefs = useRef<Array<HTMLButtonElement | null>>([]);

  function focusEnabled(fromIndex: number, direction: 1 | -1) {
    const count = items.length;
    for (let step = 1; step <= count; step++) {
      const next = (fromIndex + direction * step + count) % count;
      if (!items[next].disabled) {
        buttonRefs.current[next]?.focus();
        return;
      }
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    if (e.key === "ArrowRight") {
      e.preventDefault();
      focusEnabled(index, 1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      focusEnabled(index, -1);
    } else if (e.key === "Home") {
      e.preventDefault();
      focusEnabled(-1, 1);
    } else if (e.key === "End") {
      e.preventDefault();
      focusEnabled(items.length, -1);
    }
  }

  return (
    <div role="tablist" aria-label={label} className="flex gap-1 rounded-md bg-surface-0 p-1">
      {items.map((item, index) => {
        const isActive = item.value === value;
        return (
          <button
            key={item.value}
            ref={(el) => {
              buttonRefs.current[index] = el;
            }}
            role="tab"
            type="button"
            tabIndex={isActive ? 0 : -1}
            aria-selected={isActive}
            aria-disabled={item.disabled}
            disabled={item.disabled}
            title={item.disabled ? item.disabledReason : undefined}
            onClick={() => !item.disabled && onChange(item.value)}
            onKeyDown={(e) => handleKeyDown(e, index)}
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
