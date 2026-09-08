import { Scan, Boxes, Minimize2, Atom, Eye } from "lucide-react";
import { Surface } from "@/components/ui/Surface";

/**
 * Renamed from `PipelineVisualizer.tsx`. Step 1 used to read "Volumetric
 * MRI Ingestion — 3D sagittal / coronal DICOM & NumPy volume decoding" --
 * false. The backend (`extras/api/server.py`) explicitly marks
 * Coronal/Sagittal `available: false` for a typical single-series upload,
 * because reslicing along in-plane pixel axes is not a real anatomical
 * view. Every stage below is checked against what the pipeline actually
 * implements (`qknee/models/pipeline.py`, `README.md`'s architecture
 * table) -- no stage claims 3D reconstruction, quantum-only processing,
 * or clinical superiority.
 */
const STEPS = [
  { icon: Scan, title: "Knee MRI", detail: "2D axial DICOM series or NumPy volume, slice-by-slice" },
  { icon: Boxes, title: "Visual Feature Extraction", detail: "Frozen ResNet18 → 512-dimensional embedding per slice" },
  { icon: Minimize2, title: "Feature Compression", detail: "PCA → 4 scalars, scaled to [0, 2π] for angle encoding" },
  { icon: Atom, title: "Variational Quantum Classifier", detail: "4-qubit circuit, angle encoding + entangling layers, PennyLane simulator" },
  { icon: Eye, title: "Explainable Output", detail: "Risk score + Grad-CAM attention map on the classified slice" },
] as const;

export function ArchitectureDiagram() {
  return (
    <section className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
      <h2 className="text-lg font-semibold text-ink-primary">How it works</h2>
      <p className="mt-1 text-sm text-ink-muted">
        One forward pass, five stages. Classical feature extraction hands off to a quantum circuit simulator —
        quantum processing does not replace the classical stages, and classical baselines currently outperform
        the hybrid model on real data (see Methods).
      </p>

      <ol className="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <li key={step.title}>
              <Surface className="h-full p-4">
                <div className="flex items-center justify-between">
                  <Icon className="h-4 w-4 text-accent" aria-hidden="true" />
                  <span className="font-mono text-[10px] text-ink-faint">0{i + 1}</span>
                </div>
                <h3 className="mt-3 text-xs font-semibold text-ink-primary">{step.title}</h3>
                <p className="mt-1 text-[11px] leading-relaxed text-ink-muted">{step.detail}</p>
              </Surface>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
