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
}

export interface PredictionResponse {
  risk_score: number;
  diagnosis: string;
  gradcam_heatmap: string;
  backend: string;
  latency_ms: number | null;
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

export interface DiagnosticResult {
  riskScore: number;
  diagnosis: string;
  severity: SeverityTag;
  heatmap: string;
  backend: string;
  latencyMs: number | null;
  qubitExpectations: [number, number, number, number];
  source: "live" | "mock";
}
