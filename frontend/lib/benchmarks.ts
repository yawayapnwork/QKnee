// GENERATED — DO NOT EDIT BY HAND.
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

export const BENCHMARK_ROWS: readonly BenchmarkRow[] = [
  {
    "model": "Classical ResNet-18 (linear probe)",
    "auc": "0.6574",
    "highlight": false
  },
  {
    "model": "Classical ResNet-18 → RBF SVM",
    "auc": "0.5596",
    "highlight": false
  },
  {
    "model": "Q-Knee Hybrid (ResNet-18 → PCA(4) → 4-Qubit VQC)",
    "auc": "0.5579",
    "highlight": true
  }
] as const;
