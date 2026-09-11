import Link from "next/link";
import { ArrowRight } from "lucide-react";
import { PipelineVisual } from "@/components/landing/PipelineVisual";

/**
 * Full rebuild, not a patch of the prior Hero.tsx -- no generic
 * "revolutionizing healthcare" headline, no full-bleed gradient, no
 * decorative particle field. The headline states the actual data-flow
 * (`qknee/models/pipeline.py`'s four externally-visible stages) as the
 * hero copy itself, and `PipelineVisual` backs it with a real six-stage
 * instrument panel rather than an illustration. "What this is / is not"
 * from the previous version is preserved as the sub-line + CTA pairing,
 * not dropped -- a viewer still needs both facts before clicking.
 */
export function Hero() {
  return (
    <section id="platform" className="mx-auto max-w-6xl scroll-mt-16 px-4 py-14 sm:px-6 sm:py-20">
      <div className="grid grid-cols-1 gap-10">
        <div className="max-w-2xl">
          <p className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-accent">
            Knee MRI → Explainable Risk Triage
          </p>

          <h1 className="mt-4 font-mono text-2xl font-bold leading-tight tracking-tight text-ink-primary sm:text-3xl lg:text-4xl">
            <span className="block">KNEE MRI</span>
            <span className="block pl-3 text-ink-faint">↓ visual features</span>
            <span className="block pl-3 text-ink-faint">↓ compact quantum classifier</span>
            <span className="block pl-3 text-accent">↓ explainable risk score</span>
          </h1>

          <p className="mt-6 max-w-xl text-sm leading-relaxed text-ink-muted sm:text-base">
            Q-Knee is a hybrid classical/quantum research pipeline: a frozen ResNet18 extracts visual features from
            each MRI slice, a compression stage reduces them to four scalars, a 4-qubit variational quantum circuit
            classifies tear risk, and Grad-CAM shows which regions drove the score.
          </p>

          <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-surface-3 bg-surface-1 p-4">
              <h2 className="text-2xs font-semibold uppercase tracking-wide text-status-live">What this is</h2>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
                A research prototype for ACL/meniscal tear risk triage, evaluated on real RSNA Knee data (n=58).
              </p>
            </div>
            <div className="rounded-md border border-surface-3 bg-surface-1 p-4">
              <h2 className="text-2xs font-semibold uppercase tracking-wide text-status-fallback">
                What this is not
              </h2>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
                Not a certified medical device. Not validated for clinical use. Confirmatory radiologist review is
                required for every result.
              </p>
            </div>
          </div>

          <div className="mt-7 flex flex-col gap-2 sm:flex-row">
            <Link
              href="/workstation"
              className="inline-flex items-center justify-center gap-2 rounded-sm bg-accent px-4 py-2 text-sm font-semibold text-surface-0 transition-colors hover:bg-accent-strong"
            >
              Open Workstation
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
            <Link
              href="#pipeline"
              className="inline-flex items-center justify-center gap-2 rounded-sm bg-surface-1/60 px-4 py-2 text-sm font-medium text-accent ring-1 ring-inset ring-accent/30 transition-colors hover:bg-accent/10 hover:ring-accent/60"
            >
              Explore Architecture
            </Link>
          </div>
        </div>

        <PipelineVisual />
      </div>
    </section>
  );
}
