/**
 * The Grad-CAM disclaimer and the limitations list, written down ONCE.
 * `ExplanationWorkspace` (the on-screen workspace) and `PrintReport` (the
 * print/PDF output) both render this exact text — previously each surface
 * that wanted to state "not proof of a lesion" or "small research dataset"
 * hand-typed its own copy of the sentence, which is exactly how two
 * surfaces end up saying subtly different things about the same fact.
 */

export const GRADCAM_DISCLAIMER =
  "This is a model attention visualization — it shows which image regions most influenced the ResNet18 embedding, computed for one representative slice only. It is not a segmentation and does not prove a lesion exists; independent radiologist review is required.";

export interface Limitation {
  title: string;
  detail: string;
}

export const LIMITATIONS: readonly Limitation[] = [
  {
    title: "Small research dataset",
    detail: "Evaluated on real RSNA Knee data, n=58 studies — not large enough to establish generalization.",
  },
  {
    title: "No clinical validation",
    detail: "This has not undergone clinical trials and is not a validated diagnostic tool.",
  },
  {
    title: "Simulator-based quantum execution",
    detail: "The quantum circuit runs on PennyLane's default.qubit classical simulator, not physical quantum hardware.",
  },
  {
    title: "Research prototype",
    detail: "Investigational output only. Findings require independent review by a licensed radiologist or orthopedic clinician.",
  },
] as const;
