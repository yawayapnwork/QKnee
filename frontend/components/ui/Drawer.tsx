"use client";

import { useEffect } from "react";
import { X } from "lucide-react";

/**
 * A slide-in side panel for content that is a permanent column on wide
 * viewports but would consume too much width below `lg:` -- used by the
 * workstation's case selector (see `app/workstation/page.tsx`). Shares
 * `Modal`'s backdrop/Escape/focus-return behavior but slides from the
 * left edge instead of centering, and is explicitly a RESPONSIVE
 * accommodation, not a place to hide functionality that should just be
 * on the page.
 */
export function Drawer({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex bg-surface-0/85 animate-fade-in"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="h-full w-72 max-w-[85vw] animate-slide-in-left border-r border-surface-3 bg-surface-1 shadow-3"
      >
        <div className="flex items-center justify-between border-b border-surface-3 px-4 py-3">
          <span className="text-sm font-semibold text-ink-primary">{title}</span>
          <button onClick={onClose} aria-label="Close" className="text-ink-muted hover:text-ink-primary">
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
        <div className="overflow-y-auto">{children}</div>
      </div>
    </div>
  );
}
