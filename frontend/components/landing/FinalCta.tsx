"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowRight, LogIn, UserCheck } from "lucide-react";
import { useAuth } from "@/lib/auth-context";

const AuthModal = dynamic(() => import("@/components/auth/AuthModal").then((m) => m.AuthModal), {
  ssr: false,
  loading: () => <div className="fixed inset-0 z-50 bg-surface-0/85" aria-hidden="true" />,
});

export function FinalCta() {
  const { continueAsGuest } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);

  return (
    <section className="border-t border-surface-3">
      <div className="mx-auto flex max-w-6xl flex-col items-start gap-4 px-4 py-14 sm:px-6 sm:py-20">
        <h2 className="text-2xl font-bold tracking-tight text-ink-primary sm:text-3xl">
          Run a study through the full pipeline.
        </h2>
        <p className="max-w-xl text-sm leading-relaxed text-ink-muted">
          Try a precomputed demo case or upload a knee MRI series, and watch the ResNet18 → PCA → 4-qubit VQC →
          Grad-CAM pipeline run end to end with an explanation workspace for every result. The repository&apos;s
          n=58 evaluation numbers live separately, on{" "}
          <Link href="/methods" className="text-accent hover:underline">
            Methods
          </Link>
          .
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-3">
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
        </div>
      </div>

      {authOpen && <AuthModal open onClose={() => setAuthOpen(false)} />}
    </section>
  );
}
