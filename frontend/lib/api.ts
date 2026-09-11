import type {
  CaseSummary,
  HealthResponse,
  LoginCredentials,
  PredictionResponse,
  RegisterPayload,
  Token,
  UserProfile,
} from "./types";

// Render assigns the live service's actual hostname independently of the
// blueprint's `name:` field when a same-named service already exists, so
// this must match whatever Render shows on the service's own dashboard --
// currently https://qknee-8dv8.onrender.com. Override via
// $NEXT_PUBLIC_API_URL (see frontend/vercel.json) for any other deployment.
const rawApiUrl = (process.env.NEXT_PUBLIC_API_URL || "https://qknee-8dv8.onrender.com").trim();
export const API_BASE_URL = (rawApiUrl.length > 0 ? rawApiUrl : "https://qknee-8dv8.onrender.com").replace(/\/+$/, "");

/** Render's free tier spins the backend down after ~15 min idle. The first
 * request after that can take 30-60s to come back (a network timeout, or a
 * `/health` 503 while the process is still booting) before the backend is
 * actually reachable. Without this, a cold start looked identical to a
 * genuinely broken backend -- one failed request and the caller gave up,
 * so the workstation showed "No demo cases available" / no diagnosis
 * instead of just waiting out the boot. Retries with backoff (~50s total)
 * before surfacing a real failure. */
export async function withColdStartRetry<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  signal: AbortSignal,
  onRetry?: (attempt: number, maxAttempts: number) => void,
): Promise<T> {
  const retryDelaysMs = [3000, 5000, 8000, 13000, 20000];
  for (let attempt = 0; ; attempt++) {
    try {
      return await fn(signal);
    } catch (err) {
      if (signal.aborted || attempt >= retryDelaysMs.length) throw err;
      onRetry?.(attempt + 1, retryDelaysMs.length);
      await new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, retryDelaysMs[attempt]);
        signal.addEventListener("abort", () => {
          clearTimeout(timer);
          reject(new DOMException("Aborted", "AbortError"));
        });
      });
    }
  }
}

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

/** GET /api/v1/health (with fallback to /health) — used both for the workstation's live status badge
 * and to detect a Render free-tier cold start. Checks for status=ok or backend_ready=true. */
export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/health`, { signal, cache: "no-store" });
    if (res.ok) {
      const data = await parseJsonOrThrow<HealthResponse>(res);
      if (data.status === "ok" || data.backend_ready === true) {
        return data;
      }
      throw new ApiError(res.status, data.detail || "Backend not ready");
    }
  } catch (err) {
    if (signal?.aborted) throw err;
    // Attempt fallback to /health below if /api/v1/health failed with network/status issue
  }

  const fallbackRes = await fetch(`${API_BASE_URL}/health`, { signal, cache: "no-store" });
  const data = await parseJsonOrThrow<HealthResponse>(fallbackRes);
  if (data.status === "ok" || data.backend_ready === true) {
    return data;
  }
  throw new ApiError(fallbackRes.status, data.detail || "Backend not ready");
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

/** GET /api/v1/auth/me — proves the bearer token round-trips and returns
 * the authenticated caller's profile. Throws ApiError(401) on invalid/expired token. */
export async function fetchCurrentUser(token: string, signal?: AbortSignal): Promise<UserProfile> {
  const res = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
    signal,
    cache: "no-store",
  });
  return parseJsonOrThrow<UserProfile>(res);
}

/** POST /api/v1/predict — multipart upload of a DICOM (.dcm/.dicom) or
 * NumPy (.npy) MRI slice/volume.
 * The response's `quantum_expectations`/`n_qubits`/`quantum_backend` are the
 * VQC's own real per-qubit measurement for this request (or `null` if this
 * backend didn't run a live circuit) — see `lib/quantum-telemetry.ts`,
 * which is the only place that data should be turned into UI-facing
 * `QuantumTelemetry`. */
export async function predictScanVolume(file: File, token?: string, signal?: AbortSignal): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    headers,
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
