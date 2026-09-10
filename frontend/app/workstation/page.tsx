"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
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

// Same code-split rationale as `AppShell.tsx`'s "Sign In" trigger: most
// visits never hit the "upload without being signed in" path, so the auth
// form's JS shouldn't be part of the workstation page's initial bundle.
const AuthModal = dynamic(() => import("@/components/auth/AuthModal").then((m) => m.AuthModal), {
  ssr: false,
  loading: () => <div className="fixed inset-0 z-50 bg-surface-0/85" aria-hidden="true" />,
});
import { useAuth } from "@/lib/auth-context";
import { ApiError, fetchCase, fetchCases, fetchHealth, predictScanVolume } from "@/lib/api";
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
  return {
    riskScore: prediction.risk_score,
    diagnosis: prediction.diagnosis,
    severity: severityFromRisk(prediction.risk_score),
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
export default function WorkstationPage() {
  const { token, user, isReady } = useAuth();
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);
  const [result, setResult] = useState<DiagnosticResult | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [casesOpen, setCasesOpen] = useState(false);
  const [apiHealth, setApiHealth] = useState<ApiHealth>("checking");
  const lastFileRef = useRef<File | null>(null);
  // The in-flight `/predict` or `/api/cases/{id}` request, if any -- aborted
  // whenever a newer one supersedes it (another upload, a demo-case switch,
  // or the page unmounting) so a slow first response can never land after
  // and clobber whatever the viewer is looking at by then.
  const requestAbortRef = useRef<AbortController | null>(null);

  const canDiagnose = isReady && user?.role === "radiologist";

  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    fetchHealth(controller.signal)
      .then(() => setApiHealth("online"))
      .catch(() => setApiHealth("offline"))
      .finally(() => clearTimeout(timeout));
    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  // Fetches the real case list once, then auto-loads the first case so the
  // page never opens empty when real demo data exists. An empty list (no
  // `qknee/artifacts/real_demo_cases/` built yet) or a failed fetch just
  // leaves the page on its "No result yet" empty state -- never falls back
  // to any fabricated data.
  useEffect(() => {
    const controller = new AbortController();
    fetchCases(controller.signal)
      .then((fetched) => {
        if (controller.signal.aborted) return;
        setCases(fetched);
        if (fetched.length > 0) void loadCase(fetched[0].case_id);
      })
      .catch(() => {
        if (!controller.signal.aborted) setCases([]);
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
    setErrorMessage(null);
    try {
      const prediction = await fetchCase(caseId, controller.signal);
      if (controller.signal.aborted) return;
      setResult(toDiagnosticResult(prediction));
      setStatus("idle");
    } catch (err) {
      if (controller.signal.aborted) return;
      setErrorMessage(err instanceof ApiError ? err.detail : "Failed to load this demo case.");
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
    setStatus("loading");
    setErrorMessage(null);
    try {
      const prediction = await predictScanVolume(file, token!, controller.signal);
      if (controller.signal.aborted) return;
      setResult(toDiagnosticResult(prediction));
      setStatus("idle");
    } catch (err) {
      // A newer request superseded this one -- that request's own
      // success/error handling owns the UI now, so this stale rejection
      // (a `DOMException` named "AbortError") must render nothing.
      if (controller.signal.aborted) return;
      // Execution mandate rule 13: a failed request must never silently
      // become a mock prediction. No `DiagnosticResult` is produced here —
      // the analysis column renders `ErrorState` instead, with an
      // explicit, viewer-initiated "Load Demo Case" action rather than an
      // automatic substitution.
      setErrorMessage(
        err instanceof ApiError ? err.detail : "The Q-Knee API is unreachable (a Render cold start can take up to a minute).",
      );
      setStatus("error");
    }
  }

  function handleUpload(file: File) {
    if (!token || !canDiagnose) {
      setAuthOpen(true);
      return;
    }
    void runInference(file);
  }

  function handleRetry() {
    if (lastFileRef.current) void runInference(lastFileRef.current);
  }

  function handleLoadDemo() {
    const caseId = activeCaseId ?? cases[0]?.case_id;
    if (caseId) void loadCase(caseId);
  }

  const visibleResult = status === "error" ? null : result;
  const activeCase = cases.find((c) => c.case_id === activeCaseId) ?? null;
  const caseLabel = activeCase?.label ?? (result ? "Live Upload" : "");

  return (
    <div className="flex flex-1 flex-col">
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
      <div className="no-print grid flex-1 grid-cols-1 md:grid-cols-[minmax(0,1fr)_360px] lg:grid-cols-[220px_minmax(0,1fr)_340px] xl:grid-cols-[260px_minmax(0,1fr)_380px]">
        <aside className="hidden lg:block lg:border-r lg:border-surface-3" aria-label="Case navigation">
          <CaseNav
            cases={cases}
            activeCaseId={activeCaseId}
            onSelectCase={handleSelectCase}
            canDiagnose={Boolean(canDiagnose)}
            apiHealth={apiHealth}
            onUpload={handleUpload}
          />
        </aside>

        <Drawer open={casesOpen} onClose={() => setCasesOpen(false)} title="Demo Cases & Upload">
          <CaseNav
            cases={cases}
            activeCaseId={activeCaseId}
            onSelectCase={handleSelectCase}
            canDiagnose={Boolean(canDiagnose)}
            apiHealth={apiHealth}
            onUpload={handleUpload}
          />
        </Drawer>

        <section className="min-h-[480px] border-b border-surface-3 md:border-b-0 md:border-r" aria-label="MRI viewer">
          <MRIViewer result={visibleResult} />
        </section>

        <section className="flex flex-col divide-y divide-surface-3 [&>*]:px-4 [&>*]:py-5 sm:[&>*]:px-6" aria-label="AI analysis">
          {status === "loading" && <LoadingState />}

          {status === "error" && (
            <ErrorState message={errorMessage ?? "Unknown error."} onRetry={handleRetry} onLoadDemo={handleLoadDemo} />
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

      {authOpen && <AuthModal open onClose={() => setAuthOpen(false)} />}
    </div>
  );
}
