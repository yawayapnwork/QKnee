import { Surface } from "@/components/ui/Surface";

/**
 * States what the VQC in `qknee/models/vqc.py` actually does -- angle
 * encoding, 3 trainable RX/RY/RZ + ring-CNOT layers, Pauli-Z readout, run
 * on a classical state-vector simulator -- and just as deliberately
 * states what it does not claim. No "quantum advantage," "supremacy," or
 * "clinical superiority" language anywhere, because the repo's own
 * evaluation (see Evidence) shows the opposite on real data at n=58.
 */
const FACTS = [
  { label: "Qubits", value: "4" },
  { label: "Variational layers", value: "3" },
  { label: "Encoding", value: "RY angle encoding, one qubit per PCA scalar" },
  { label: "Entanglement", value: "Ring of CNOT gates per layer" },
  { label: "Readout", value: "Pauli-Z expectation → linear + sigmoid" },
  { label: "Execution", value: "PennyLane default.qubit simulator (classical hardware)" },
] as const;

export function QuantumSection() {
  return (
    <section className="border-t border-surface-3">
      <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6 sm:py-20">
        <p className="font-mono text-xs font-semibold uppercase tracking-[0.2em] text-accent">Quantum Component</p>
        <h2 className="mt-3 text-2xl font-bold tracking-tight text-ink-primary sm:text-3xl">
          What the quantum circuit actually does
        </h2>

        <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="space-y-4 text-sm leading-relaxed text-ink-muted">
            <p>
              The four PCA-compressed scalars are encoded as RY rotation angles, one per qubit. Three trainable
              variational layers follow: each applies an independent RX/RY/RZ rotation to every qubit, then a ring
              of CNOT gates entangles neighboring qubits. The circuit is measured in the Pauli-Z basis, and those
              four expectation values feed a small classical linear layer with a sigmoid to produce the final risk
              probability.
            </p>
            <p>
              This runs on PennyLane&apos;s <code className="text-ink-primary">default.qubit</code> state-vector
              simulator — a classical simulation of a quantum circuit, not physical quantum hardware.
            </p>
            <div className="rounded-md border border-warning/30 bg-warning/10 px-4 py-3 text-xs font-medium text-warning">
              This page makes no claim of quantum advantage, quantum supremacy, or clinical superiority. On the real
              evaluation set (n=58), the hybrid VQC currently trails both classical baselines — see{" "}
              <a href="#evidence" className="underline">
                Evidence
              </a>
              .
            </div>
          </div>

          <Surface className="min-w-0 p-4">
            <h3 className="text-2xs font-semibold uppercase tracking-wide text-ink-faint">Circuit parameters</h3>
            <dl className="mt-3 divide-y divide-surface-3">
              {FACTS.map((fact) => (
                <div key={fact.label} className="py-2.5 text-xs">
                  <dt className="text-ink-faint">{fact.label}</dt>
                  <dd className="mt-1 break-words font-mono text-ink-primary">{fact.value}</dd>
                </div>
              ))}
            </dl>
          </Surface>
        </div>
      </div>
    </section>
  );
}
