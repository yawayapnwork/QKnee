import type {
  HealthResponse,
  LoginCredentials,
  PredictionResponse,
  RegisterPayload,
  Token,
} from "./types";

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_URL || "https://qknee.onrender.com").replace(/\/$/, "");

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
 * NumPy (.npy) MRI slice/volume. Requires a `radiologist` bearer token. */
export async function predictScanVolume(file: File, token: string): Promise<PredictionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  });
  return parseJsonOrThrow<PredictionResponse>(res);
}
