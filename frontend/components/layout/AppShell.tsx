"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LogOut } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";

// Code-split: every route mounts `AppShell` (it's in the root layout), but
// only a viewer who actually clicks "Sign In" needs the auth form's JS
// (Field/SelectField, the login/register request logic). `ssr:false`
// because it's a client-only overlay with no content to server-render
// before the click that reveals it -- there is nothing to hydrate-mismatch
// against. `loading` renders the same backdrop the loaded modal will use
// so a slow connection sees an immediate, deliberate "opening" state
// instead of the click appearing to do nothing.
const AuthModal = dynamic(() => import("@/components/auth/AuthModal").then((m) => m.AuthModal), {
  ssr: false,
  loading: () => <div className="fixed inset-0 z-50 bg-surface-0/85" aria-hidden="true" />,
});

const NAV_LINKS = [
  { href: "/workstation", label: "Workstation" },
  { href: "/methods", label: "Methods" },
] as const;

/**
 * The one persistent navigation surface across all three pages (Landing,
 * Workstation, Methods) -- replaces the old, separate `Navbar.tsx`
 * (landing-only) and the auth/case-switcher logic duplicated inside
 * `CommandBar.tsx`. Flat wordmark, no gradient tile.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const { user, signOut, isReady } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);
  const pathname = usePathname();

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-surface-3 bg-surface-0/95">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <Link href="/" className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-md bg-accent text-xs font-bold text-surface-0">
              QK
            </span>
            <span className="font-semibold tracking-tight text-ink-primary">Q-Knee</span>
          </Link>

          <nav aria-label="Primary" className="flex items-center gap-1 sm:gap-4">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                aria-current={pathname === link.href ? "page" : undefined}
                className={cn(
                  "rounded-md px-2 py-1.5 text-sm font-medium transition-colors",
                  pathname === link.href
                    ? "text-ink-primary"
                    : "text-ink-muted hover:text-ink-primary",
                )}
              >
                {link.label}
              </Link>
            ))}

            {isReady && user ? (
              <div className="ml-2 flex items-center gap-3">
                <span className="hidden text-xs text-ink-muted sm:inline">
                  {user.full_name} · <span className="text-accent">{user.role}</span>
                </span>
                <Button variant="tertiary" size="sm" onClick={signOut} aria-label="Sign out">
                  <LogOut className="h-3.5 w-3.5" aria-hidden="true" />
                  <span className="hidden sm:inline">Sign Out</span>
                </Button>
              </div>
            ) : (
              <Button variant="secondary" size="sm" className="ml-2" onClick={() => setAuthOpen(true)}>
                Sign In
              </Button>
            )}
          </nav>
        </div>
      </header>

      <main className="flex flex-1 flex-col">{children}</main>

      {/* Mounted only once actually opened -- `dynamic()` above triggers
          its chunk fetch on first mount, so gating the mount on `authOpen`
          (rather than always rendering it with `open={false}`) is what
          makes the code-split real instead of eager on every route. */}
      {authOpen && <AuthModal open onClose={() => setAuthOpen(false)} />}
    </div>
  );
}
