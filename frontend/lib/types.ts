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
  /** Unified prediction-provenance category — see `qknee.observability.provenance.classify`
   * (the Python module both this API and the Streamlit dashboard derive their provenance from).
   * One of "live" | "precomputed_demo" | "mock_fallback" | "cached" | "proxy". This is the
   * authoritative signal the UI must badge/gate on — never `backend` (above), which is a
   * free-form legacy tag. A real forward pass through untrained/randomly-initialized weights
   * (AUDIT.md B3/C4b) is reported here as "mock_fallback", never "live", even though `backend`
   * still says "live" for that case. */
  provenance: Provenance;
  /** Exact display string for `provenance` (e.g. "LIVE", "MOCK/FALLBACK") — render verbatim. */
  provenance_label: string;
  /** "trained_checkpoint" | "random_fallback" when a real model forward pass ran; `null` when no
   * model ran at all (mock/precomputed-demo/proxy paths). */
  model_source: ModelSource | null;
  /** Exact display string for `model_source`, or `null` iff `model_source` is `null`. */
  model_source_label: string | null;
  /** "quantum_simulator" when a real PennyLane circuit executed for this response, else
   * "unavailable". Never labeled "SIMULATION MODE" — the quantum simulator running is this
   * project's normal, expected state (AUDIT.md C4c); only "unavailable" means something is
   * actually missing. */
  quantum_execution: QuantumExecution;
  /** Exact display string for `quantum_execution` — render verbatim. */
  quantum_execution_label: string;
}

export type Provenance = "live" | "precomputed_demo" | "mock_fallback" | "cached" | "proxy";
export type ModelSource = "trained_checkpoint" | "random_fallback";
/** The backend's own wire vocabulary for `PredictionResponse.quantum_execution` — the API only
 * ever reports one of these two values, for a request it actually served. */
export type QuantumExecution = "quantum_simulator" | "unavailable";
/**
 * App-level extension of `QuantumExecution`, used only on `ProvenanceInfo.quantumExecution` —
 * never on the wire type above. Adds `"not_verifiable"`: real per-qubit numbers are shown (a
 * precomputed demo case's stored `qubitExpectations`), but no live circuit execution can be
 * attested by the frontend for *this specific result*, so it must not render with the same
 * confident "QUANTUM SIMULATOR" badge a genuine backend-attested execution gets. The backend
 * itself never sends this value — only `provenanceForPreset` (`lib/provenance.ts`) constructs it,
 * for demo/preset results exclusively.
 */
export type AppQuantumExecution = QuantumExecution | "not_verifiable";

/** App-level provenance bundle threaded onto every `DiagnosticResult` — see `lib/provenance.ts`.
 * Built exactly once per result (from a live `PredictionResponse` or explicitly for a preset/demo
 * case), never re-derived at a render call site, so the badge a viewer sees always matches the
 * actual execution path. */
export interface ProvenanceInfo {
  provenance: Provenance;
  provenanceLabel: string;
  modelSource: ModelSource | null;
  modelSourceLabel: string | null;
  quantumExecution: AppQuantumExecution;
  quantumExecutionLabel: string;
  /** False whenever this result must never be presented as an ordinary successful "LIVE" result
   * (currently: `provenance === "mock_fallback"`) — the UI must render an unmistakable, high-
   * contrast badge in that case, per AUDIT.md P1 #5 requirement 5. */
  isTrustworthy: boolean;
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
  /** Per-head checkpoint availability — "available" | "unavailable" — e.g.
   * `{"primary": "available", "acl": "unavailable", ...}`. Was present on
   * the backend response but had no TypeScript representation at all;
   * surfaced on the Methods page's Model Status panel. */
  model_status: Record<string, string>;
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
 * Raw per-qubit circuit output — nothing else. Whether this data is
 * trustworthy, live, demo, or fabricated is answered ENTIRELY by the
 * parent `DiagnosticResult.provenance.quantumExecution` — this interface
 * used to carry its own separate provenance field (`QuantumTelemetryProvenance`,
 * a second, hyphenated vocabulary independent of `Provenance`); that field
 * is deleted. Two provenance systems for one product is exactly the
 * "competing vocabularies" defect the provenance model exists to prevent.
 * `expectations` is empty iff `provenance.quantumExecution === "unavailable"`.
 */
export interface QuantumTelemetry {
  expectations: number[];
  nQubits: number | null;
  device: string | null;
}

/**
 * One diagnostic result, one provenance. `provenance: ProvenanceInfo` is
 * the ONLY field any component may branch on to decide whether this result
 * is live, demo, mock, cached, or proxied — never `backend` (a raw,
 * free-form debug string, kept only for the Technical Details panel) and
 * never a second `source`/telemetry-specific field (deleted; this struct
 * used to carry three overlapping "what kind of result is this" fields).
 */
export interface DiagnosticResult {
  riskScore: number;
  diagnosis: string;
  severity: SeverityTag;
  /** Raw backend tag (e.g. "live", "mock", "cache-fallback/case_003") —
   * diagnostic/debug value only. Never branch UI behavior on this; use
   * `provenance` instead. */
  backend: string;
  latencyMs: number | null;
  quantumTelemetry: QuantumTelemetry;
  volume: VolumeView;
  provenance: ProvenanceInfo;
}
