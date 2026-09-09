import Link from "next/link";

/**
 * Landing-page-only sub-nav. AppShell already provides the sitewide
 * sticky header (wordmark + Workstation/Methods + auth) on every route --
 * this is not a replacement for that, it's the page-specific orientation
 * bar the brief calls for: the full "Q-KNEE / Hybrid Quantum MRI Research
 * Platform" lockup plus in-page section jumps and the two landing CTAs.
 * Anchors target the section ids declared by Hero (#platform),
 * ArchitectureSection (#pipeline), and EvidenceSection (#evidence);
 * "Workstation" and "Open Workstation" both go to the real route.
 */
const SECTION_LINKS = [
  { href: "#platform", label: "Platform" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#evidence", label: "Evidence" },
  { href: "/workstation", label: "Workstation" },
] as const;

export function LandingNav() {
  return (
    <div className="border-b border-surface-3">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-sm font-bold tracking-[0.2em] text-ink-primary">Q-KNEE</span>
          <span className="hidden text-xs text-ink-faint sm:inline">Hybrid Quantum MRI Research Platform</span>
        </div>

        <nav aria-label="Landing sections" className="flex flex-wrap items-center gap-5">
          {SECTION_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="text-xs font-medium text-ink-muted transition-colors hover:text-ink-primary"
            >
              {link.label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link
            href="#pipeline"
            className="hidden rounded-sm px-3 py-1.5 text-xs font-medium text-ink-muted ring-1 ring-inset ring-surface-4 transition-colors hover:bg-surface-2 hover:text-ink-primary sm:inline-flex"
          >
            Explore Architecture
          </Link>
          <Link
            href="/workstation"
            className="inline-flex items-center rounded-sm bg-accent px-3 py-1.5 text-xs font-semibold text-surface-0 transition-colors hover:bg-accent-strong"
          >
            Open Workstation
          </Link>
        </div>
      </div>
    </div>
  );
}
