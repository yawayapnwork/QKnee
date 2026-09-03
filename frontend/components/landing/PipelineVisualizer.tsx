import { Boxes, Scan, Atom, Flame } from "lucide-react";
import { Card } from "@/components/ui/Card";

const STEPS = [
  {
    icon: Scan,
    title: "Volumetric MRI Ingestion",
    detail: "3D sagittal / coronal DICOM & NumPy volume decoding",
    tone: "teal" as const,
  },
  {
    icon: Boxes,
    title: "ResNet-18 Spatial Extractor",
    detail: "512-dim embedding compressed to 4 orthogonal scalars",
    tone: "cyan" as const,
  },
  {
    icon: Atom,
    title: "4-Qubit Variational Circuit",
    detail: "Angle-encoding RY rotations + CNOT entanglement",
    tone: "teal" as const,
  },
  {
    icon: Flame,
    title: "Pauli-Z + Grad-CAM Overlay",
    detail: "⟨Z⟩ expectation output with layer-4 spatial saliency",
    tone: "cyan" as const,
  },
];

const toneRing = {
  teal: "ring-teal-500/40 text-teal-400",
  cyan: "ring-cyan-400/40 text-cyan-400",
};

export function PipelineVisualizer() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-20">
      <div className="mb-12 text-center">
        <h2 className="text-2xl font-bold text-slate-100 sm:text-3xl">Hybrid Inference Pipeline</h2>
        <p className="mt-2 text-sm text-slate-500">
          One forward pass, four stages — classical feature extraction handed off to a NISQ simulator.
        </p>
      </div>

      <div className="relative grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <svg
          className="pointer-events-none absolute inset-0 hidden h-full w-full lg:block"
          preserveAspectRatio="none"
        >
          <line
            x1="12.5%"
            y1="50%"
            x2="87.5%"
            y2="50%"
            stroke="url(#pipeline-gradient)"
            strokeWidth="2"
            strokeDasharray="6 6"
          />
          <defs>
            <linearGradient id="pipeline-gradient" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#14b8a6" />
              <stop offset="100%" stopColor="#22d3ee" />
            </linearGradient>
          </defs>
        </svg>

        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <Card key={step.title} className="relative z-10 p-6 transition-transform hover:-translate-y-1">
              <div className="flex items-start justify-between">
                <div className={`flex h-11 w-11 items-center justify-center rounded-xl bg-slate-800 ring-2 ${toneRing[step.tone]}`}>
                  <Icon className="h-5 w-5" />
                </div>
                <span className="font-mono text-xs text-slate-600">0{i + 1}</span>
              </div>
              <h3 className="mt-4 text-sm font-semibold text-slate-100">{step.title}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{step.detail}</p>
            </Card>
          );
        })}
      </div>
    </section>
  );
}
