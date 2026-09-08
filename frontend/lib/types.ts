export const ROLES = ["radiologist", "researcher", "clinical_auditor"] as const;
export type Role = (typeof ROLES)[number];

export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  role: Role;
  created_at: string;
  is_active: boolean;
}

export interface Token {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
  user: UserProfile;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterPayload {
  email: string;
  password: string;
  full_name: string;
  role?: Role;
  invite_code?: string;
}

export interface PredictionResponse {
  risk_score: number;
  diagnosis: string;
  gradcam_heatmap: string;
  backend: string;
  latency_ms: number | null;
  /** Raw per-qubit Pauli-Z expectation values in [-1, 1], read directly from the
   * executed VQC circuit before the classical readout collapses them into
   * `risk_score` (see `qknee.models.pipeline.PipelineRunner.get_pauli_z_expectations`).
   * `null` when this response's backend didn't run a quantum circuit for this
   * request (e.g. `backend === "mock"`) — never a fallback/placeholder array. */
  quantum_expectations: number[] | null;
  /** Length of `quantum_expectations`. `null` exactly when `quantum_expectations` is `null`. */
  n_qubits: number | null;
  /** PennyLane device identifier the circuit ran on (e.g. "default.qubit").
   * `null` exactly when `quantum_expectations` is `null`. */
  quantum_backend: string | null;
  /** Base64-encoded PNG of the RAW MRI slice actually classified — no Grad-CAM,
   * no overlay of any kind. Always a different asset from `gradcam_overlay`. */
  base_image: string;
  /** Base64-encoded PNG (RGBA, transparent) of ONLY the Grad-CAM heatmap, sized
   * to match `base_image` — composite over `base_image` client-side at a
   * caller-controlled opacity. `null` when no heatmap was computed. */
  gradcam_overlay: string | null;
  /** Which plane `gradcam_overlay` belongs to. `null` iff `gradcam_overlay` is `null`. */
  gradcam_plane: AnatomicalPlane | null;
  /** 0-based index, within `planes[gradcam_plane].slices`, of the exact slice
   * `gradcam_overlay` was computed for — Grad-CAM exists for one representative
   * slice only, never every slice. `null` iff `gradcam_overlay` is `null`. */
  gradcam_slice_index: number | null;
  /** Real per-plane volume metadata, keyed by plane — see `RawPlaneInfo`. Wire format
   * (snake_case field names), straight off the API — see `PlaneInfo`/`toVolumeView` in
   * `lib/viewer.ts` for the camelCase app-level shape derived from this. */
  planes: Record<AnatomicalPlane, RawPlaneInfo>;
  /** The plane `risk_score`/`gradcam_overlay` were computed from. */
  primary_plane: AnatomicalPlane;
  /** 0-based index, within `planes[primary_plane].slices`, of the slice actually classified. */
  primary_slice_index: number;
}

export type AnatomicalPlane = "axial" | "coronal" | "sagittal";

/**
 * One anatomical plane's real, backend-reported availability — never a
 * claim the backend can't back up. `available: false` means the backend
 * genuinely does not produce this plane for this upload (most commonly:
 * only "axial" — the volume's native slice-stacking axis — is ever real
 * for a typical single-series MRI upload; reslicing along the in-plane
 * pixel axes isn't a real anatomical Coronal/Sagittal view). The UI must
 * disable an unavailable plane, never pretend it works. Wire format
 * (snake_case), as returned by `PredictionResponse.planes`.
 */
export interface RawPlaneInfo {
  available: boolean;
  num_slices: number;
  /** Base64-encoded PNGs, index-aligned with `num_slices`. Empty when `available` is false. */
  slices: string[];
}

/** App-level (camelCase) counterpart of `RawPlaneInfo` — see `lib/viewer.ts`. */
export interface PlaneInfo {
  available: boolean;
  numSlices: number;
  slices: string[];
}

/** App-level (camelCase) per-result volume/viewer state, derived from a
 * `PredictionResponse` (`lib/viewer.ts#volumeViewFromPrediction`) or a
 * `PresetCase` (`lib/viewer.ts#volumeViewFromPreset`) — never assembled by
 * hand at a call site, so a live result's viewer data can't accidentally
 * mix in preset/other-result data (mirrors `quantum-telemetry.ts`'s
 * single-call-site pattern for the same reason). */
export interface VolumeView {
  planes: Record<AnatomicalPlane, PlaneInfo>;
  primaryPlane: AnatomicalPlane;
  primarySliceIndex: number;
  gradcamOverlay: string | null;
  gradcamPlane: AnatomicalPlane | null;
  gradcamSliceIndex: number | null;
}

export interface HealthResponse {
  status: string;
  backend_ready: boolean;
  detail?: string | null;
  mode: string;
  user_store_backend: string;
  cache_backend: string;
  artifacts: Record<string, boolean>;
  quantum_simulator: {
    available: boolean;
    device?: string;
    n_qubits?: number;
    pennylane_version?: string;
    reason?: string;
  };
  latency_benchmark?: {
    model: string;
    latency_ms_per_sample: number;
    roc_auc: number;
  } | null;
}

export type SeverityTag = "Normal" | "Indeterminate" | "Urgent Surgical Consult";

export interface PresetCase {
  id: string;
  label: string;
  description: string;
  category: "ACL Tear" | "Intact Meniscus" | "Multi-Compartment Defect";
  riskScore: number;
  qubitExpectations: [number, number, number, number];
}

/**
 * Where a `QuantumTelemetry` reading came from — determines what label the
 * UI is allowed to show alongside it (never silently mixed):
 *   - "live": this exact upload's own executed VQC circuit
 *     (`PredictionResponse.backend === "live"`).
 *   - "precomputed-demo": real Pauli-Z measurements, but recorded ahead of
 *     time for a demo/preset case rather than this request's own upload
 *     (a frontend `PresetCase`, or the API's `cache-fallback/...` backend).
 *   - "unavailable": no real per-qubit data exists for this result. The UI
 *     must show "Quantum telemetry unavailable", never a fake/derived value.
 */
export type QuantumTelemetryProvenance = "live" | "precomputed-demo" | "unavailable";

export interface QuantumTelemetry {
  provenance: QuantumTelemetryProvenance;
  /** Empty when provenance is "unavailable". */
  expectations: number[];
  nQubits: number | null;
  device: string | null;
}

export interface DiagnosticResult {
  riskScore: number;
  diagnosis: string;
  severity: SeverityTag;
  backend: string;
  latencyMs: number | null;
  quantumTelemetry: QuantumTelemetry;
  volume: VolumeView;
  source: "live" | "mock";
}
