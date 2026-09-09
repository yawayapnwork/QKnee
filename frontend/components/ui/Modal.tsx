"use client";

import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * The one modal shell in the product -- backdrop, `role="dialog"`,
 * `aria-modal`, Escape-to-close, backdrop-click-to-close. `AuthModal` used
 * to hand-roll all of this itself; any future modal (there is currently
 * exactly one consumer) gets this behavior for free instead of a second,
 * slightly-different reimplementation.
 *
 * Focus management: opening moves focus onto the dialog's close button
 * (the one control guaranteed to exist regardless of form state) so
 * screen-reader users land inside the dialog rather than having focus
 * silently stay on a now-hidden trigger; closing restores focus to
 * whatever element had it before the dialog opened, so keyboard users
 * aren't dropped back at the top of the page.
 */
export function Modal({
  open,
  onClose,
  titleId,
  title,
  children,
  className,
}: {
  open: boolean;
  onClose: () => void;
  titleId: string;
  title: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}) {
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const previouslyFocused = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  useEffect(() => {
    if (!open) return;
    previouslyFocused.current = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();
    // Cleanup (not an `open`-dependent branch) so focus is restored both
    // when a parent keeps this mounted and flips `open` back to false,
    // and when a parent conditionally unmounts it entirely on close (e.g.
    // a lazily-loaded consumer that only mounts the modal while open) --
    // React runs this on unmount either way.
    return () => {
      previouslyFocused.current?.focus();
    };
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-surface-0/85 p-4 animate-fade-in"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={cn("w-full max-w-md rounded-lg border border-surface-3 bg-surface-1 shadow-3", className)}
      >
        <div className="flex items-center justify-between border-b border-surface-3 px-6 py-4">
          <h2 id={titleId} className="font-semibold text-ink-primary">
            {title}
          </h2>
          <button ref={closeButtonRef} onClick={onClose} aria-label="Close dialog" className="text-ink-muted hover:text-ink-primary">
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
