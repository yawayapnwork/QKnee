import { Scan, Boxes, Minimize2, Atom, Eye } from "lucide-react";
import { Surface } from "@/components/ui/Surface";

/**
 * Replaces `ArchitectureDiagram.tsx` (deleted). Same source-of-truth
 * discipline that file already established -- every stage name and
 * dimension below is checked against `qknee/models/pipeline.py`,
 * `qknee/models/vqc.py`, and `qknee/config/config.yaml`
 * (n_qubits=4, n_layers=3) -- but goes one step further into how each
 * stage actually works, since this section is the "Architecture" the
 * nav/CTA promise, not a five-icon teaser.
 */
const STAGES = [
  {
    icon: Scan,
    title: "1. Knee MRI",
    detail:
      "A 2D axial DICOM series (or NumPy volume) is decoded slice-by-slice. No 3D reconstruction and no multi-plane reslicing are performed — a single-series upload is processed exactly as acquired.",
  },
  {
    icon: Boxes,
    title: "2. ResNet18 feature extraction",
    detail:
      "A frozen, ImageNet-pretrained ResNet18 backbone converts each slice into a 512-dimensional embedding. The backbone is not fine-tuned during training — only the layers downstream of it are.",
  },
  {
    icon: Minimize2,
    title: "3. Feature compression",
    detail:
      "A fitted PCA (or an equivalent bottleneck) reduces the 512-dim embedding to 4 scalars — one per qubit — then rescales them into [0, 2π] so they can be used as rotation angles.",
  },
  {
    icon: Atom,
    title: "4. 4-qubit VQC",
    detail:
      "Each scalar becomes an RY rotation angle on one qubit. Three trainable variational layers follow, each applying RX/RY/RZ rotations per qubit and a ring of CNOT gates, executed on PennyLane's default.qubit state-vector simulator.",
  },
  {
    icon: Eye,
    title: "5. Risk score + Grad-CAM",
    detail:
      "The circuit's four Pauli-Z expectation values pass through a small linear + sigmoid readout to produce a risk probability. Grad-CAM is computed separately on the ResNet18 backbone to show which image regions most influenced the embedding.",
  },
] as const;

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

        <ol className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {STAGES.map((stage) => {
            const Icon = stage.icon;
            return (
              <li key={stage.title}>
                <Surface className="h-full p-4">
                  <Icon className="h-4 w-4 text-accent" aria-hidden="true" />
                  <h3 className="mt-3 text-xs font-semibold text-ink-primary">{stage.title}</h3>
                  <p className="mt-1.5 text-2xs leading-relaxed text-ink-muted">{stage.detail}</p>
                </Surface>
              </li>
            );
          })}
        </ol>
      </div>
    </section>
  );
}
