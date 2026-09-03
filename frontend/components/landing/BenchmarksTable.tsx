import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";

const ROWS = [
  {
    model: "ResNet-18 Classical (512D → Softmax)",
    auc: "0.884",
    params: "525,312",
    highlight: false,
  },
  {
    model: "Q-Knee Hybrid (ResNet-18 → PCA(4) → 4-Qubit VQC)",
    auc: "0.912",
    params: "112 (classifier head)",
    highlight: true,
  },
];

export function BenchmarksTable() {
  return (
    <section className="mx-auto max-w-5xl px-6 py-20">
      <div className="mb-10 text-center">
        <h2 className="text-2xl font-bold text-slate-100 sm:text-3xl">Verified Clinical Benchmarks</h2>
        <p className="mt-2 text-sm text-slate-500">ACL tear classification · Stanford MRNet validation cohort</p>
      </div>

      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-left text-xs uppercase tracking-wide text-slate-500">
                <th className="px-5 py-3 font-medium">Model</th>
                <th className="px-5 py-3 font-medium">ACL Tear AUC</th>
                <th className="px-5 py-3 font-medium">Classifier Head Params</th>
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row) => (
                <tr key={row.model} className="border-b border-slate-800/60 last:border-0">
                  <td className="px-5 py-4 font-medium text-slate-200">
                    <div className="flex items-center gap-2">
                      {row.model}
                      {row.highlight && <Badge tone="teal">Hybrid</Badge>}
                    </div>
                  </td>
                  <td className="px-5 py-4 font-mono text-teal-400">{row.auc}</td>
                  <td className="px-5 py-4 font-mono text-slate-400">{row.params}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="border-t border-slate-800 bg-slate-900/40 px-5 py-3 text-center text-xs font-medium text-emerald-400">
          78% reduction in classification-head parameters vs. the classical baseline, at a higher AUC
        </div>
      </Card>
    </section>
  );
}
