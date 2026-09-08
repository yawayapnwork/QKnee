import { ArchitectureDiagram } from "@/components/landing/ArchitectureDiagram";
import { BenchmarksTable } from "@/components/methods/BenchmarksTable";
import { ModelStatusPanel } from "@/components/methods/ModelStatusPanel";
import { MethodsSection } from "@/components/methods/MethodsSection";

/**
 * The technical surface (FRONTEND_REDESIGN.md's new information
 * architecture): architecture, quantum circuit, evaluation, explainability,
 * limitations, deployment -- everything a skeptical reviewer needs to
 * verify a claim, kept off the landing page entirely.
 */
export default function MethodsPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6">
      <h1 className="text-2xl font-bold text-ink-primary">Methods</h1>
      <p className="mt-2 text-sm text-ink-muted">
        Architecture, evaluation, and known limitations — presented plainly, including the results that don&apos;t
        favor the quantum component.
      </p>

      <MethodsSection title="Architecture">
        <ArchitectureDiagram />
        <p>
          Multi-slice volumes are mean/attention/top-k pooled per slice through the same 2D ResNet18 — this is
          slice-wise feature extraction and pooling, not 3D convolution or volumetric reconstruction. Only the
          axial plane (the volume&apos;s native slice-stacking axis) is ever reported as a real anatomical view;
          Coronal/Sagittal are marked unavailable for a typical single-series upload rather than fabricated by
          reslicing in-plane pixel axes.
        </p>
      </MethodsSection>

      <MethodsSection title="Quantum circuit">
        <p>
          4 qubits, continuous angle encoding (RX then RY per qubit) of the PCA-compressed features, followed by
          3 variational layers (RX/RY/RZ rotations + a CNOT entangling ring), measuring Pauli-Z expectation
          values via PennyLane&apos;s <code className="text-ink-primary">default.qubit</code> state-vector simulator.
        </p>
        <p>
          This is a genuine NISQ simulator execution, not a physical QPU — no physical-hardware claim is made
          anywhere in this project. A simulator backend does not make the model&apos;s inference any less real: the
          circuit executes, and its measured output feeds the classical readout that produces the risk score.
        </p>
        <p className="text-ink-primary">
          Quantum processing does not replace the classical stages. ResNet18 feature extraction and PCA
          compression are classical; only the final 4-scalar classification step is quantum.
        </p>
      </MethodsSection>

      <MethodsSection title="Evaluation">
        <BenchmarksTable />
        <p>
          The classical ResNet-18 linear probe outperforms both the SVM baseline and the hybrid VQC on real
          data. No quantum advantage is claimed or demonstrated at this sample size (n=58, a single held-out
          split) — see <code className="text-ink-primary">RESULTS.md</code> for full per-condition numbers and
          the Effusion label-quality exclusion.
        </p>
        <ModelStatusPanel />
      </MethodsSection>

      <MethodsSection title="Explainability">
        <p>
          Grad-CAM on ResNet18&apos;s final convolutional block highlights image regions that most influenced the
          feature embedding, computed for one representative slice per study (never every slice). This is a
          model-attention visualization, not a segmentation and not a lesion detector — it does not confirm a
          finding and requires independent radiologist interpretation.
        </p>
      </MethodsSection>

      <MethodsSection title="Limitations">
        <ul className="list-disc space-y-1.5 pl-5">
          <li>Not a certified medical device; not validated for clinical use.</li>
          <li>Evaluated on n=58 real studies — a single train/test split, not cross-validated.</li>
          <li>Hybrid VQC underperforms classical baselines on the available real-data evaluation.</li>
          <li>Coronal/Sagittal views are unavailable for typical single-series uploads (see Architecture).</li>
          <li>Some model heads (see Model checkpoint status above) have no trained checkpoint on a given deployment and are reported unavailable rather than producing a score.</li>
        </ul>
      </MethodsSection>

      <MethodsSection title="Deployment & execution model">
        <p>
          Two interfaces share one ML pipeline (<code className="text-ink-primary">qknee.models.pipeline.PipelineRunner</code>):
          this Next.js/FastAPI pair (secondary, exploratory) calls it over HTTP; a Streamlit dashboard (primary,
          judged interface) calls it in-process. Both derive provenance and checkpoint status from the same
          shared modules — see <code className="text-ink-primary">ARCHITECTURE.md</code> at the repository root
          for the full picture.
        </p>
      </MethodsSection>
    </div>
  );
}
