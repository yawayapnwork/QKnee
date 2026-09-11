"use client";

import { useEffect, useRef, useState } from "react";
import { StudyHeader } from "@/components/workstation/StudyHeader";
import { CaseNav } from "@/components/workstation/CaseNav";
import { MRIViewer } from "@/components/workstation/MRIViewer";
import { PredictionPanel } from "@/components/workstation/PredictionPanel";
import { QuantumTelemetry } from "@/components/workstation/QuantumTelemetry";
import { ExplanationPanel } from "@/components/workstation/ExplanationPanel";
import { StatusBar } from "@/components/workstation/StatusBar";
import { PrintReport } from "@/components/workstation/PrintReport";
import { LoadingState } from "@/components/shared/LoadingState";
import { ErrorState } from "@/components/shared/ErrorState";
import { EmptyState } from "@/components/shared/EmptyState";
import { Drawer } from "@/components/ui/Drawer";
import { ApiError, fetchCase, fetchCases, fetchHealth, predictScanVolume, withColdStartRetry } from "@/lib/api";
import { severityFromRisk } from "@/lib/severity";
import { quantumTelemetryFromPrediction } from "@/lib/quantum-telemetry";
import { provenanceFromPrediction } from "@/lib/provenance";
import { volumeViewFromPrediction } from "@/lib/viewer";
import type { CaseSummary, DiagnosticResult, PredictionResponse } from "@/lib/types";

type Status = "idle" | "loading" | "error";
type ApiHealth = "checking" | "online" | "offline";

/** Builds this page's one `DiagnosticResult` shape from a `PredictionResponse`
 * — used identically whether that response came from a live upload
 * (`predictScanVolume`) or a real, offline-scored demo case (`fetchCase`).
 * There is no second, "preset" conversion path: both sources return the
 * exact same wire shape, so both go through the exact same conversion. */
function toDiagnosticResult(prediction: PredictionResponse): DiagnosticResult {
  const severityThresholds = {
    normalMax: prediction.severity_band_normal_max,
    urgentMin: prediction.severity_band_urgent_min,
  };
  return {
    riskScore: prediction.risk_score,
    diagnosis: prediction.diagnosis,
    reason: prediction.reason,
    severity: severityFromRisk(prediction.risk_score, severityThresholds),
    severityThresholds,
    backend: prediction.backend,
    latencyMs: prediction.latency_ms,
    quantumTelemetry: quantumTelemetryFromPrediction(prediction),
    volume: volumeViewFromPrediction(prediction),
    provenance: provenanceFromPrediction(prediction),
  };
}

/**
 * Full rebuild of the workstation shell (not a cosmetic pass over the
 * prior grid) around a real 5-region desktop layout: a TOP identity/
 * provenance/actions bar, a LEFT case-navigation column, a CENTER imaging
 * surface that dominates the available space, a RIGHT AI-analysis column
 * (divided by rules, not stacked cards), and a BOTTOM technical-status
 * strip. Below `lg:` the left column moves into a `Drawer`; below `md:`
 * center and right stack vertically into a single scrollable analysis
 * flow — see the grid template below.
 *
 * `apiHealth` is fetched exactly once, here, and threaded down to both
 * `StudyHeader` (the reachability indicator) and `CaseNav` (whether the
 * upload control may honestly say "Live Analysis") — one fetch, one
 * source of truth, instead of each component polling `/health` on its
 * own and risking two different answers on screen at once.
 *
 * Demo cases: `GET /api/cases` (real RSNA Knee studies, scored once
 * offline by `scripts/build_real_demo_cases.py`) replaces the old
 * hand-authored fake-case constant entirely. Selecting a demo case calls
 * `GET /api/cases/{id}` and runs its response through the exact same
 * `toDiagnosticResult` conversion a live upload uses — there is no
 * separate fabricated-data code path left in this page.
 */
export type WorkstationErrorKind =
  | "initialization"
  | "file_selection"
  | "upload_request"
  | "inference";

export interface WorkstationError {
  kind: WorkstationErrorKind;
  title: string;
  message: string;
  detail?: string;
}

export default function WorkstationPage() {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [resultSource, setResultSource] = useState<"demo" | "live" | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [activeError, setActiveError] = useState<WorkstationError | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [casesOpen, setCasesOpen] = useState(false);
  const [apiHealth, setApiHealth] = useState<ApiHealth>("checking");
  const lastFileRef = useRef<File | null>(null);
  // The in-flight `/predict` or `/api/cases/{id}` request, if any -- aborted
  // whenever a newer one supersedes it (another upload, a demo-case switch,
  // or the page unmounting) so a slow first response can never land after
  // and clobber whatever the viewer is looking at by then.
  const requestAbortRef = useRef<AbortController | null>(null);

  const casesLoadedRef = useRef(false);
  const apiOnlineRef = useRef(false);

  const markApiOnline = () => {
    apiOnlineRef.current = true;
    setApiHealth("online");
  };

  // Poll health and keep status updated; automatically reload cases when backend comes online
  useEffect(() => {
    let isMounted = true;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    let activeHealthAbort: AbortController | null = null;

    async function checkHealthAndSync() {
      activeHealthAbort?.abort();
      const controller = new AbortController();
      activeHealthAbort = controller;

      // Allow up to 20s per check to tolerate free-tier Render backend cold starts
      const timeout = setTimeout(() => controller.abort(), 20000);

      try {
        const health = await fetchHealth(controller.signal);
        clearTimeout(timeout);
        if (!isMounted) return;

        if (health.status === "ok" || health.backend_ready) {
          markApiOnline();

          // If demo cases have not loaded yet (e.g. backend was cold starting when page mounted), fetch them now
          if (!casesLoadedRef.current) {
            try {
              const fetched = await fetchCases(controller.signal);
              if (isMounted && fetched.length > 0) {
                casesLoadedRef.current = true;
                setCases(fetched);
                setActiveError((prev) => (prev?.kind === "initialization" ? null : prev));
                void loadCase(fetched[0].case_id);
              }
            } catch {
              // Retry cases on subsequent health cycle
            }
          }

          // Periodic heartbeat every 30s while online
          pollTimer = setTimeout(() => void checkHealthAndSync(), 30000);
        } else {
          // Prevent an old/stale failed health-check request from overwriting a newer successful "online" state
          if (!apiOnlineRef.current) {
            setApiHealth("offline");
          }
          // Fast retry every 5s while offline
          pollTimer = setTimeout(() => void checkHealthAndSync(), 5000);
        }
      } catch {
        clearTimeout(timeout);
        if (!isMounted) return;
        // Prevent an old/stale failed health-check request from overwriting a newer successful "online" state
        if (!apiOnlineRef.current) {
          setApiHealth("offline");
        }
        // Fast retry every 5s while offline
        pollTimer = setTimeout(() => void checkHealthAndSync(), 5000);
      }
    }

    void checkHealthAndSync();

    return () => {
      isMounted = false;
      if (pollTimer) clearTimeout(pollTimer);
      activeHealthAbort?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetches the real case list once, then auto-loads the first case so the
  // page never opens empty when real demo data exists. Retries through a
  // Render free-tier cold start (`withColdStartRetry`) instead of giving up
  // after one failed request -- an empty list only means the fetch never
  // succeeded even after ~50s of retrying, never a fabricated fallback.
  useEffect(() => {
    const controller = new AbortController();
    withColdStartRetry((signal) => fetchCases(signal), controller.signal)
      .then((fetched) => {
        if (controller.signal.aborted) return;
        casesLoadedRef.current = true;
        setCases(fetched);
        if (fetched.length > 0) {
          markApiOnline();
          void loadCase(fetched[0].case_id);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          setCases([]);
          setActiveError({
            kind: "initialization",
            title: "Workstation Initialization Failed",
            message: "Unable to load demo cases. The Q-Knee API may be offline or starting up.",
          });
          setStatus("error");
        }
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cancel any in-flight request on unmount -- otherwise its `.then`/`.catch`
  // could still fire after the page is gone.
  useEffect(() => {
    return () => requestAbortRef.current?.abort();
  }, []);

  async function loadCase(caseId: string) {
    requestAbortRef.current?.abort();
    const controller = new AbortController();
    requestAbortRef.current = controller;
    setActiveCaseId(caseId);
    setStatus("loading");
    setActiveError(null);
    try {
      const prediction = await fetchCase(caseId, controller.signal);
      if (controller.signal.aborted) return;
      markApiOnline();
      setResult(toDiagnosticResult(prediction));
      setResultSource("demo");
      setStatus("idle");
    } catch (err) {
      if (controller.signal.aborted) return;
      setActiveError({
        kind: "initialization",
        title: "Failed to Load Demo Case",
        message: err instanceof ApiError ? err.detail : "Failed to load this demo case.",
      });
      setStatus("error");
    }
  }

  function handleSelectCase(caseId: string) {
    void loadCase(caseId);
    setCasesOpen(false);
  }

  async function runInference(file: File) {
    lastFileRef.current = file;
    requestAbortRef.current?.abort();
    const controller = new AbortController();
    requestAbortRef.current = controller;
    setIsUploading(true);
    setStatus("loading");
    setActiveError(null);
    try {
      const prediction = await predictScanVolume(file, undefined, controller.signal);
      if (controller.signal.aborted) return;
      setResult(toDiagnosticResult(prediction));
      setResultSource("live");
      setActiveCaseId(null);
      setStatus("idle");
      setIsUploading(false);
      markApiOnline();
    } catch (err) {
      // A newer request superseded this one -- that request's own
      // success/error handling owns the UI now, so this stale rejection
      // (a `DOMException` named "AbortError") must render nothing.
      if (controller.signal.aborted) return;
      setIsUploading(false);

      if (err instanceof ApiError) {
        if (err.status >= 500 || err.status === 404) {
          // Server / network failure
          setActiveError({
            kind: "upload_request",
            title: "Upload Request Failed",
            message: "The Q-Knee server returned an error during upload. If the server is cold-starting, please retry in a moment.",
            detail: err.detail,
          });
        } else if (err.status === 401 || err.status === 403) {
          // Backend service limitation
          setActiveError({
            kind: "upload_request",
            title: "Inference Service Unavailable",
            message: "Live model inference is currently restricted by the backend service.",
            detail: err.detail,
          });
        } else {
          // 400 / 422 - Inference input processing failure
          setActiveError({
            kind: "inference",
            title: "Model Inference Failed",
            message: err.detail || "The model could not process this scan volume.",
            detail: err.detail,
          });
        }
      } else {
        // Network / connection timeout
        setActiveError({
          kind: "upload_request",
          title: "API Unreachable",
          message: "The Q-Knee API is unreachable or timed out (a Render cold start can take up to a minute). Please retry.",
        });
      }
      setStatus("error");
    }
  }

  function handleUpload(file: File) {
    if (!file) return;

    // 1. File-selection check: format validation (.dcm, .dicom, .npy)
    const fileName = file.name.toLowerCase();
    const isValidExt = fileName.endsWith(".dcm") || fileName.endsWith(".dicom") || fileName.endsWith(".npy");
    if (!isValidExt) {
      setActiveError({
        kind: "file_selection",
        title: "Unsupported File Format",
        message: `"${file.name}" is not a supported format. Please upload a DICOM (.dcm, .dicom) or NumPy (.npy) volume.`,
      });
      setStatus("error");
      return;
    }

    // 2. File-selection check: non-empty file
    if (file.size === 0) {
      setActiveError({
        kind: "file_selection",
        title: "Empty File",
        message: `The selected file "${file.name}" is empty (0 bytes). Please upload a valid MRI scan volume.`,
      });
      setStatus("error");
      return;
    }

    setActiveError(null);
    void runInference(file);
  }

  function handleRetry() {
    if (lastFileRef.current) {
      void runInference(lastFileRef.current);
    } else if (activeCaseId) {
      void loadCase(activeCaseId);
    }
  }

  function handleLoadDemo() {
    const caseId = activeCaseId ?? cases[0]?.case_id;
    if (caseId) {
      setActiveError(null);
      void loadCase(caseId);
    }
  }

  const visibleResult = status === "error" ? null : result;
  const activeCase = cases.find((c) => c.case_id === activeCaseId) ?? null;
  const caseLabel = activeCase?.label ?? (resultSource === "live" ? "Live Upload" : result ? "Live Upload" : "");

  return (
    <div className="flex flex-1 flex-col min-h-0 md:h-full md:overflow-hidden">
      <StudyHeader
        caseLabel={caseLabel}
        result={visibleResult}
        apiHealth={apiHealth}
        onOpenCases={() => setCasesOpen(true)}
      />

      {/* Column widths step twice, not once, across the 3-column range --
          `lg` (1024px, where the case-nav column first appears) uses
          narrower fixed columns than `xl` (1280px). At a flat 260px+380px
          the imaging viewer -- the single most important region on this
          screen -- would be squeezed to ~384px right at 1024px, the exact
          width the brief calls out as a floor. Widening back up at `xl`
          keeps the more generous columns for viewports that have the
          room to spare. */}
      <div className="no-print grid flex-1 min-h-0 grid-cols-1 md:grid-cols-[minmax(0,1fr)_360px] lg:grid-cols-[220px_minmax(0,1fr)_340px] xl:grid-cols-[260px_minmax(0,1fr)_380px] md:grid-rows-[minmax(0,1fr)]">
        <aside className="relative hidden lg:flex lg:flex-col min-h-0 overflow-y-auto lg:border-r lg:border-surface-3" aria-label="Case navigation">
          <CaseNav
            cases={cases}
            activeCaseId={activeCaseId}
            onSelectCase={handleSelectCase}
            apiHealth={apiHealth}
            onUpload={handleUpload}
            isUploading={isUploading}
          />
        </aside>

        <Drawer open={casesOpen} onClose={() => setCasesOpen(false)} title="Demo Cases & Upload">
          <CaseNav
            cases={cases}
            activeCaseId={activeCaseId}
            onSelectCase={handleSelectCase}
            apiHealth={apiHealth}
            onUpload={handleUpload}
            isUploading={isUploading}
          />
        </Drawer>

        <section className="flex flex-col min-h-[420px] md:min-h-0 overflow-hidden border-b border-surface-3 md:border-b-0 md:border-r" aria-label="MRI viewer">
          <MRIViewer result={visibleResult} />
        </section>

        <section className="flex flex-col min-h-0 md:overflow-y-auto divide-y divide-surface-3 [&>*]:px-4 [&>*]:py-5 sm:[&>*]:px-6" aria-label="AI analysis">
          {status === "loading" && <LoadingState />}

          {status === "error" && (
            <ErrorState
              title={activeError?.title}
              message={activeError?.message ?? "An unexpected error occurred."}
              onRetry={
                activeError?.kind !== "file_selection" &&
                (lastFileRef.current || activeCaseId)
                  ? handleRetry
                  : undefined
              }
              onLoadDemo={cases.length > 0 ? handleLoadDemo : undefined}
            />
          )}

          {status === "idle" && result && (
            <>
              <PredictionPanel result={result} />
              <QuantumTelemetry
                telemetry={result.quantumTelemetry}
                provenance={result.provenance}
                latencyMs={result.latencyMs}
              />
              <ExplanationPanel result={result} />
            </>
          )}

          {status === "idle" && !result && (
            <EmptyState title="No result yet" description="Select a demo case or upload a scan to begin." />
          )}
        </section>
      </div>

      <StatusBar result={visibleResult} />

      {/* The one thing `window.print()` (see `ReportExport.tsx`) actually
          shows -- hidden on screen, revealed only under `@media print`
          (see `.print-only` in `app/globals.css`). Rendered only when
          there is a real result to report; an error/empty state has
          nothing honest to print. */}
      {visibleResult && <PrintReport result={visibleResult} caseLabel={caseLabel} />}
    </div>
  );
}
