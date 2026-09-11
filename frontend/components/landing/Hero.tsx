"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowRight, LogIn, UserCheck } from "lucide-react";
import { PipelineVisual } from "@/components/landing/PipelineVisual";
import { useAuth } from "@/lib/auth-context";

const AuthModal = dynamic(() => import("@/components/auth/AuthModal").then((m) => m.AuthModal), {
  ssr: false,
  loading: () => <div className="fixed inset-0 z-50 bg-surface-0/85" aria-hidden="true" />,
});

/**
 * Full rebuild, not a patch of the prior Hero.tsx -- no generic
 * "revolutionizing healthcare" headline, no full-bleed gradient, no
 * decorative particle field. The headline states the actual data-flow
 * (`qknee/models/pipeline.py`'s four externally-visible stages) as the
 * hero copy itself, and `PipelineVisual` backs it with a real six-stage
 * instrument panel rather than an illustration. Dual-mode entry provides
 * both instant unauthenticated exploration ('Continue as Guest') and
 * authenticated clinician session management ('Sign In').
 */
export function Hero() {
  const { continueAsGuest } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);

  return (
    <section id="platform" className="mx-auto max-w-6xl scroll-mt-16 px-4 py-14 sm:px-6 sm:py-20">
      <div className="grid grid-cols-1 gap-10">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-2 rounded-full border border-accent/25 bg-accent-subtle px-3 py-1 text-xs font-semibold text-accent-strong">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden="true" />
            Clinical Knee MRI · Explainable Risk Triage
          </div>

          <h1 className="mt-4 text-3xl font-bold tracking-tight text-ink-primary sm:text-4xl lg:text-5xl">
            Precision knee MRI analysis with transparent decision support.
          </h1>

          <p className="mt-4 max-w-xl text-sm leading-relaxed text-ink-secondary sm:text-base">
            Q-Knee is a hybrid classical/quantum research pipeline: a frozen ResNet18 extracts visual features from
            each MRI slice, a compression stage reduces them to four scalars, a 4-qubit variational quantum circuit
            classifies tear risk, and Grad-CAM maps show which anatomical regions drove the score.
          </p>

          <div className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div className="rounded-md border border-surface-3 bg-white p-4 shadow-xs">
              <h2 className="text-2xs font-semibold uppercase tracking-wide text-status-live">What this is</h2>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
                A research prototype for ACL/meniscal tear risk triage, evaluated on real RSNA Knee data (n=58).
              </p>
            </div>
            <div className="rounded-md border border-surface-3 bg-white p-4 shadow-xs">
              <h2 className="text-2xs font-semibold uppercase tracking-wide text-status-fallback">
                What this is not
              </h2>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">
                Not a certified medical device. Not validated for clinical use. Confirmatory radiologist review is
                required for every result.
              </p>
            </div>
          </div>

          <div className="mt-7 flex flex-wrap items-center gap-3">
            <Link
              href="/workstation"
              onClick={continueAsGuest}
              className="inline-flex items-center justify-center gap-2 rounded-md bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-xs transition-colors hover:bg-accent-strong focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <UserCheck className="h-4 w-4" aria-hidden="true" />
              Continue as Guest
              <ArrowRight className="h-4 w-4" aria-hidden="true" />
            </Link>
            <button
              type="button"
              onClick={() => setAuthOpen(true)}
              className="inline-flex items-center justify-center gap-2 rounded-md border border-surface-3 bg-white px-5 py-2.5 text-sm font-semibold text-ink-primary shadow-xs transition-colors hover:bg-surface-2 hover:border-surface-4"
            >
              <LogIn className="h-4 w-4 text-ink-muted" aria-hidden="true" />
              Sign In
            </button>
            <Link
              href="#pipeline"
              className="inline-flex items-center justify-center gap-1.5 px-3 py-2.5 text-sm font-medium text-ink-muted hover:text-ink-primary transition-colors"
            >
              Explore Architecture
            </Link>
          </div>
        </div>

        <PipelineVisual />
      </div>

      {authOpen && <AuthModal open onClose={() => setAuthOpen(false)} />}
    </section>
  );
}
