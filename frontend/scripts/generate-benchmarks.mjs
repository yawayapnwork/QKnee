#!/usr/bin/env node
// Regenerates lib/benchmarks.ts from the canonical
// qknee/artifacts/kaggle_benchmark_summary.json artifact (produced by
// qknee/models/evaluate.py). Run after re-running that evaluation:
//
//   node scripts/generate-benchmarks.mjs
//
// This is Option B from the remediation brief: a generated typed data
// module, not a runtime fetch (the artifact isn't served by any API this
// frontend calls) and not a second hand-copied literal (the previous
// defect). `lib/__tests__/benchmark-sync.test.ts` re-reads the artifact
// directly and fails the build if the generated file's numbers ever
// diverge from it -- whether or not someone remembered to rerun this
// script -- so drift is caught even if this file is hand-edited.

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(__dirname, "..");
const ARTIFACT_PATH = join(FRONTEND_ROOT, "..", "qknee", "artifacts", "kaggle_benchmark_summary.json");
const OUTPUT_PATH = join(FRONTEND_ROOT, "lib", "benchmarks.ts");

// UI display label + highlight flag for each model the artifact reports.
// The "excl_effusion" alternate score is used (not the plain final_score)
// because Effusion ground-truth labels are audited as unreliable (see
// RESULTS.md §2 / the artifact's own `exclusion_reasons`) -- BenchmarksTable
// has always shown the 9-condition, Effusion-excluded number.
const MODEL_MAP = [
  { key: "Baseline Classical ResNet18", label: "Classical ResNet-18 (linear probe)", highlight: false },
  { key: "Classical ResNet18 + RBF SVM", label: "Classical ResNet-18 → RBF SVM", highlight: false },
  { key: "Hybrid Q-Knee VQC", label: "Q-Knee Hybrid (ResNet-18 → PCA(4) → 4-Qubit VQC)", highlight: true },
];

const artifact = JSON.parse(readFileSync(ARTIFACT_PATH, "utf-8"));

const rows = MODEL_MAP.map(({ key, label, highlight }) => {
  const model = artifact.models[key];
  if (!model) throw new Error(`Model "${key}" not found in ${ARTIFACT_PATH}`);
  const score = model.alternate_scores?.excl_effusion?.final_score;
  if (typeof score !== "number") {
    throw new Error(`Model "${key}" has no alternate_scores.excl_effusion.final_score in ${ARTIFACT_PATH}`);
  }
  return { model: label, auc: score.toFixed(4), highlight };
});

const output = `// GENERATED — DO NOT EDIT BY HAND.
// Regenerate with: node scripts/generate-benchmarks.mjs
// Source: qknee/artifacts/kaggle_benchmark_summary.json (qknee/models/evaluate.py),
// the "excl_effusion" alternate score — see RESULTS.md §2 for why Effusion is excluded.
// lib/__tests__/benchmark-sync.test.ts re-reads that artifact directly and fails if
// these numbers ever drift from it, whether or not this script was rerun.

export interface BenchmarkRow {
  model: string;
  auc: string;
  highlight: boolean;
}

export const BENCHMARK_ROWS: readonly BenchmarkRow[] = ${JSON.stringify(rows, null, 2)} as const;
`;

writeFileSync(OUTPUT_PATH, output);
console.log(`Wrote ${OUTPUT_PATH} from ${ARTIFACT_PATH}`);
for (const row of rows) console.log(`  ${row.auc}  ${row.model}${row.highlight ? "  [highlight]" : ""}`);
