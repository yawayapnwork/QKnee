import type {
  CaseSummary,
  HealthResponse,
  LoginCredentials,
  PredictionResponse,
  RegisterPayload,
  Token,
} from "./types";

// AUDIT.md P1 #9: matches extras/deployment/render.yaml's `name: qknee-api`
// (Render derives the service URL from that field as `<name>.onrender.com`)
// -- the old default here (`qknee.onrender.com`) named a Render service
// that render.yaml never actually configures. Override via
// $NEXT_PUBLIC_API_URL (see frontend/vercel.json) for any other deployment.
export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "https://qknee-api.onrender.com").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? body);
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/** GET /api/v1/health — used both for the workstation's live status badge
 * and to detect a Render free-tier cold start (a timeout/network failure)
 * so the UI can fall back to preset/mock data instead of hanging. */
export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/health`, { signal, cache: "no-store" });
  return parseJsonOrThrow<HealthResponse>(res);
}

export async function loginClinician(credentials: LoginCredentials): Promise<Token> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(credentials),
  });
  return parseJsonOrThrow<Token>(res);
}

export async function registerClinician(data: RegisterPayload): Promise<Token> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  await parseJsonOrThrow(res);
  // /register returns the created profile, not a token — log the clinician
  // in immediately afterward so registration doubles as sign-in.
  return loginClinician({ username: data.email, password: data.password });
}

/** POST /api/v1/predict — multipart upload of a DICOM (.dcm/.dicom) or
 * NumPy (.npy) MRI slice/volume. Requires a `radiologist` bearer token.
 * The response's `quantum_expectations`/`n_qubits`/`quantum_backend` are the
 * VQC's own real per-qubit measurement for this request (or `null` if this
 * backend didn't run a live circuit) — see `lib/quantum-telemetry.ts`,
 * which is the only place that data should be turned into UI-facing
 * `QuantumTelemetry`. */
export async function predictScanVolume(file: File, token: string, signal?: AbortSignal): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
    signal,
  });
  return parseJsonOrThrow<PredictionResponse>(res);
}

/** GET /api/cases — real RSNA Knee studies scored once offline (see
 * `scripts/build_real_demo_cases.py`), never fabricated preset data.
 * Public, no auth required — matches `CaseNav`'s "demo cases available to
 * everyone without signing in" policy. Returns `[]` (not an error) if the
 * backend hasn't built its case cache yet. */
export async function fetchCases(signal?: AbortSignal): Promise<CaseSummary[]> {
  const res = await fetch(`${API_BASE_URL}/api/cases`, { signal, cache: "no-store" });
  return parseJsonOrThrow<CaseSummary[]>(res);
}

/** GET /api/cases/{case_id} — the full prediction payload for one real case
 * from `fetchCases()`, in the exact same shape `predictScanVolume` returns
 * for a live upload. Its `provenance` comes back `"precomputed_demo"`
 * (never `"live"`): real model, real quantum circuit, computed once
 * offline and replayed verbatim, not this request's own inference. */
export async function fetchCase(caseId: string, signal?: AbortSignal): Promise<PredictionResponse> {
  const res = await fetch(`${API_BASE_URL}/api/cases/${encodeURIComponent(caseId)}`, { signal, cache: "no-store" });
  return parseJsonOrThrow<PredictionResponse>(res);
}
