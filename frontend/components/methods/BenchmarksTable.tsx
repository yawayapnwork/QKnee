import { Badge } from "@/components/ui/Badge";
import { Table, TableHead, Th, Tr, Td } from "@/components/ui/Table";
import { BENCHMARK_ROWS } from "@/lib/benchmarks";

// `lib/benchmarks.ts` is generated from qknee/artifacts/kaggle_benchmark_summary.json
// (qknee/models/evaluate.py) by scripts/generate-benchmarks.mjs -- not a
// hand-typed literal here. Real RSNA Knee ground truth, n=58 studies.
// Effusion excluded per RESULTS.md §2 (an audited label-quality defect).
// Directional, small-sample results -- not a clinical validation, and the
// hybrid VQC trails both classical baselines, stated plainly rather than
// hidden. `lib/__tests__/benchmark-sync.test.ts` re-reads the artifact
// directly and fails the build if these numbers ever drift from it.
const ROWS = BENCHMARK_ROWS;

/**
 * Rendered inside `app/methods/page.tsx`'s `<MethodsSection title="Evaluation">`
 * -- this component supplies the table/disclaimer body only, never its own
 * top-level heading (a duplicate "Evaluation" heading was a real defect
 * caught during design-system verification: `MethodsSection` already
 * provides the section's `<h2>`).
 */
export function BenchmarksTable() {
  return (
    <div>
      <p className="text-sm text-ink-muted">
        Real RSNA Knee ground truth, n=58 studies · macro-averaged ROC-AUC, 9 conditions (Effusion excluded — see
        RESULTS.md §2) · directional, small-sample, not a clinical validation.
      </p>

      <div className="mt-4">
        <Table caption="Macro-averaged ROC-AUC by model, RSNA Knee n=58">
          <TableHead>
            <Th>Model</Th>
            <Th>Macro-AUC (9 conditions)</Th>
          </TableHead>
          <tbody>
            {ROWS.map((row) => (
              <Tr key={row.model}>
                <Td header>
                  <div className="flex items-center gap-2">
                    {row.model}
                    {row.highlight && <Badge tone="accent">Hybrid</Badge>}
                  </div>
                </Td>
                <Td className="font-mono text-ink-primary">{row.auc}</Td>
              </Tr>
            ))}
          </tbody>
        </Table>
        <p className="mt-2 rounded-sm border border-warning/30 bg-warning/10 px-4 py-2 text-xs font-medium text-warning">
          The hybrid VQC trails both classical baselines on real data at this sample size (n=58) — no quantum
          advantage is demonstrated. See <code className="text-ink-primary">RESULTS.md</code> and{" "}
          <code className="text-ink-primary">qknee/artifacts/kaggle_benchmark_summary.json</code> for the full
          per-condition breakdown and methodology.
        </p>
      </div>
    </div>
  );
}
