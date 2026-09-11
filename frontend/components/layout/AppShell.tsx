"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV_LINKS = [
  { href: "/workstation", label: "Workstation" },
  { href: "/methods", label: "Methods" },
] as const;

/**
 * The one persistent navigation surface across all three pages (Landing,
 * Workstation, Methods) -- flat wordmark, no gradient tile, clean public
 * application navigation.
 */
export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  const isWorkstation = pathname === "/workstation";

  return (
    <div className={cn("flex min-h-screen flex-col", isWorkstation && "md:h-dvh md:max-h-dvh md:overflow-hidden")}>
      <header className="no-print sticky top-0 z-40 shrink-0 border-b border-surface-3 bg-white/95 backdrop-blur-sm shadow-xs">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          <Link href="/" className="group flex items-center gap-2">
            <span className="text-base font-semibold tracking-tight text-ink-primary transition-colors group-hover:text-accent-strong">
              <span className="text-accent">Q</span>Knee
            </span>
          </Link>

          <nav aria-label="Primary" className="flex items-center gap-1 sm:gap-4">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                aria-current={pathname === link.href ? "page" : undefined}
                className={cn(
                  "rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors",
                  pathname === link.href
                    ? "bg-accent-subtle font-semibold text-accent-strong"
                    : "text-ink-secondary hover:bg-surface-2 hover:text-ink-primary",
                )}
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className={cn("flex flex-1 flex-col", isWorkstation && "md:min-h-0 md:overflow-hidden")}>{children}</main>
    </div>
  );
}
