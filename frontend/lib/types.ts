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
  heatmap: string;
  backend: string;
  latencyMs: number | null;
  quantumTelemetry: QuantumTelemetry;
  source: "live" | "mock";
}
