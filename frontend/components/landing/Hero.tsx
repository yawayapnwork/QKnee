import Link from "next/link";
import { ArrowRight } from "lucide-react";

/**
 * Rebuilt from scratch (FRONTEND_REDESIGN.md Part Q). No full-viewport
 * hero, no gradient wordmark, no glow CTA, no unhedged headline sitting
 * above a subordinate disclaimer. "What this is" and "what this is not"
 * carry equal visual weight, side by side, because both are load-bearing
 * facts a viewer needs before clicking anywhere.
 */
export function Hero() {
  return (
    <section className="mx-auto max-w-4xl px-4 py-12 sm:px-6 sm:py-16">
      <h1 className="text-2xl font-bold tracking-tight text-ink-primary sm:text-3xl">Q-Knee</h1>
      <p className="mt-2 max-w-2xl text-sm leading-relaxed text-ink-muted sm:text-base">
        A hybrid classical/quantum research pipeline for knee MRI triage: ResNet18 feature extraction, PCA
        compression, and a 4-qubit variational quantum classifier, with Grad-CAM explanation.
      </p>

      <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2">
        <div className="rounded-lg border border-surface-3 bg-surface-1 p-4">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-status-live">What this is</h2>
          <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
            A research prototype for ACL/meniscal tear risk triage, evaluated on real RSNA Knee data (n=58).
          </p>
        </div>
        <div className="rounded-lg border border-surface-3 bg-surface-1 p-4">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-status-fallback">What this is not</h2>
          <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
            Not a certified medical device. Not validated for clinical use. Confirmatory radiologist review
            is required for every result.
          </p>
        </div>
      </div>

      <div className="mt-6 flex flex-col gap-2 sm:flex-row">
        <Link
          href="/workstation"
          className="inline-flex items-center justify-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-semibold text-surface-0 hover:bg-accent-muted hover:text-ink-primary"
        >
          Open Workstation
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
        <Link
          href="/methods"
          className="inline-flex items-center justify-center gap-2 rounded-md bg-surface-2 px-4 py-2 text-sm font-medium text-ink-primary ring-1 ring-inset ring-surface-3 hover:bg-surface-3"
        >
          Explore Methods
        </Link>
      </div>
    </section>
  );
}
