import Link from "next/link";
import { ArrowRight } from "lucide-react";

export function FinalCta() {
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
        <Link
          href="/workstation"
          className="mt-2 inline-flex items-center justify-center gap-2 rounded-sm bg-accent px-5 py-2.5 text-sm font-medium text-white shadow-xs transition-colors hover:bg-accent-strong"
        >
          Open the workstation
          <ArrowRight className="h-4 w-4" aria-hidden="true" />
        </Link>
      </div>
    </section>
  );
}
