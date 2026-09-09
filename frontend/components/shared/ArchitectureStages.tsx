import { Scan, Boxes, Minimize2, Atom, Eye } from "lucide-react";
import { Surface } from "@/components/ui/Surface";

/**
 * The ONE five-stage pipeline description in the product. Used to exist
 * twice -- `ArchitectureDiagram.tsx` (rendered on `/methods`) and
 * `ArchitectureSection.tsx` (rendered on the landing page) each carried
 * their own independently-maintained copy of this array, and
 * `ArchitectureSection.tsx`'s own header comment falsely claimed the first
 * one had been "deleted" when it was still live and imported. Every stage
 * name/dimension here is checked against `qknee/models/pipeline.py`,
 * `qknee/models/vqc.py`, and `qknee/config/config.yaml` (n_qubits=4,
 * n_layers=3).
 */
export const ARCHITECTURE_STAGES = [
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

/**
 * Just the icon-card grid -- no heading, no intro copy, no top-level
 * `<section>`. Callers (the landing page's `ArchitectureSection`, the
 * Methods page's "Architecture" `MethodsSection`) each own their own
 * heading/prose/spacing scale; this component only ever renders once per
 * page and never duplicates a heading its caller already provides.
 */
export function ArchitectureStageGrid() {
  return (
    <ol className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
      {ARCHITECTURE_STAGES.map((stage) => {
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
  );
}
