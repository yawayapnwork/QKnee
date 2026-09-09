import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { BENCHMARK_ROWS } from "../benchmarks";

const FRONTEND_ROOT = join(__dirname, "..", "..");
const ARTIFACT_PATH = join(FRONTEND_ROOT, "..", "qknee", "artifacts", "kaggle_benchmark_summary.json");

/** UI display label -> canonical artifact model key. Must stay in sync
 * with `scripts/generate-benchmarks.mjs`'s own `MODEL_MAP`. */
const MODEL_MAP: Record<string, string> = {
  "Classical ResNet-18 (linear probe)": "Baseline Classical ResNet18",
  "Classical ResNet-18 → RBF SVM": "Classical ResNet18 + RBF SVM",
  "Q-Knee Hybrid (ResNet-18 → PCA(4) → 4-Qubit VQC)": "Hybrid Q-Knee VQC",
};

/**
 * The regression test the hostile review explicitly demanded: not "does
 * the citation text exist" (the older `benchmark-claims.test.ts` already
 * checked that), but "do the displayed NUMBERS still equal the canonical
 * artifact's own numbers." This reads `kaggle_benchmark_summary.json`
 * directly -- the same file `scripts/generate-benchmarks.mjs` reads to
 * produce `lib/benchmarks.ts` -- so a stale generated file fails here
 * whether or not someone remembered to rerun the generator, and a
 * hand-edited `lib/benchmarks.ts` fails here too.
 */
describe("hostile-review regression: benchmark UI values cannot silently drift from the canonical artifact", () => {
  const artifact = JSON.parse(readFileSync(ARTIFACT_PATH, "utf-8"));

  it("BENCHMARK_ROWS is non-empty and every row has a known artifact mapping", () => {
    expect(BENCHMARK_ROWS.length).toBeGreaterThan(0);
    for (const row of BENCHMARK_ROWS) {
      expect(MODEL_MAP[row.model], `no artifact-key mapping for displayed model "${row.model}"`).toBeDefined();
    }
  });

  it("every displayed AUC exactly matches the artifact's excl_effusion final_score, to 4 decimal places", () => {
    for (const row of BENCHMARK_ROWS) {
      const artifactKey = MODEL_MAP[row.model];
      const canonicalScore = artifact.models?.[artifactKey]?.alternate_scores?.excl_effusion?.final_score;
      expect(typeof canonicalScore, `artifact has no excl_effusion.final_score for "${artifactKey}"`).toBe("number");
      expect(row.auc, `"${row.model}" displays ${row.auc} but the artifact currently reports ${canonicalScore.toFixed(4)}`).toBe(
        canonicalScore.toFixed(4),
      );
    }
  });

  it("exactly one row (the hybrid VQC) is highlighted", () => {
    const highlighted = BENCHMARK_ROWS.filter((r) => r.highlight);
    expect(highlighted).toHaveLength(1);
    expect(highlighted[0]?.model).toMatch(/Hybrid/);
  });

  it("the hybrid VQC does not outperform either classical baseline (the honest, currently-true finding this UI exists to state)", () => {
    const hybrid = BENCHMARK_ROWS.find((r) => r.highlight);
    const baselines = BENCHMARK_ROWS.filter((r) => !r.highlight);
    expect(hybrid).toBeDefined();
    for (const baseline of baselines) {
      expect(Number(hybrid!.auc)).toBeLessThanOrEqual(Number(baseline.auc));
    }
  });
});
