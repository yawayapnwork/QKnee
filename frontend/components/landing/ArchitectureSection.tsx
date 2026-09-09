import { ArchitectureStageGrid } from "@/components/shared/ArchitectureStages";

/**
 * The landing page's architecture section. Stage data/cards live in
 * `ArchitectureStageGrid` (`components/shared/ArchitectureStages.tsx`) --
 * the SAME component `/methods`' "Architecture" section renders, so there
 * is exactly one five-stage pipeline description in the product, not two
 * independently-maintained copies that can silently disagree.
 */
export function ArchitectureSection() {
  return (
    <section id="pipeline" className="scroll-mt-16 border-t border-surface-3">
      <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6 sm:py-20">
        <p className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-accent">Architecture</p>
        <h2 className="mt-3 text-2xl font-bold tracking-tight text-ink-primary sm:text-3xl">
          One forward pass, five stages
        </h2>
        <p className="mt-3 max-w-2xl text-sm leading-relaxed text-ink-muted">
          Classical feature extraction hands off to a quantum circuit simulator — quantum processing replaces one
          stage of the pipeline, not the whole thing, and classical baselines currently outperform the hybrid model
          on real data (see <a href="#evidence" className="text-accent hover:underline">Evidence</a>).
        </p>

        <div className="mt-8">
          <ArchitectureStageGrid />
        </div>
      </div>
    </section>
  );
}
