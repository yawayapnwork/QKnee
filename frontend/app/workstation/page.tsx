"use client";

import { useEffect, useRef, useState } from "react";
import { StudyHeader } from "@/components/workstation/StudyHeader";
import { CaseNav } from "@/components/workstation/CaseNav";
import { MRIViewer } from "@/components/workstation/MRIViewer";
import { PredictionPanel } from "@/components/workstation/PredictionPanel";
import { QuantumTelemetry } from "@/components/workstation/QuantumTelemetry";
import { ExplanationPanel } from "@/components/workstation/ExplanationPanel";
import { StatusBar } from "@/components/workstation/StatusBar";
import { LoadingState } from "@/components/shared/LoadingState";
import { ErrorState } from "@/components/shared/ErrorState";
import { EmptyState } from "@/components/shared/EmptyState";
import { AuthModal } from "@/components/auth/AuthModal";
import { Drawer } from "@/components/ui/Drawer";
import { useAuth } from "@/lib/auth-context";
import { ApiError, fetchHealth, predictScanVolume } from "@/lib/api";
import { PRESET_CASES, mockDiagnosticResult, severityFromRisk } from "@/lib/mock-data";
import { quantumTelemetryFromPrediction } from "@/lib/quantum-telemetry";
import { provenanceFromPrediction } from "@/lib/provenance";
import { volumeViewFromPrediction } from "@/lib/viewer";
import type { DiagnosticResult, PresetCase } from "@/lib/types";

type Status = "idle" | "loading" | "error";
type ApiHealth = "checking" | "online" | "offline";

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
 */
export default function WorkstationPage() {
  const { token, user, isReady } = useAuth();
  const [activeCase, setActiveCase] = useState<PresetCase>(PRESET_CASES[0]);
  const [result, setResult] = useState<DiagnosticResult | null>(() => mockDiagnosticResult(PRESET_CASES[0]));
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [casesOpen, setCasesOpen] = useState(false);
  const [apiHealth, setApiHealth] = useState<ApiHealth>("checking");
  const lastFileRef = useRef<File | null>(null);

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

  function handleSelectCase(preset: PresetCase) {
    setActiveCase(preset);
    setStatus("idle");
    setErrorMessage(null);
    setResult(mockDiagnosticResult(preset));
    setCasesOpen(false);
  }

  async function runInference(file: File) {
    lastFileRef.current = file;
    setStatus("loading");
    setErrorMessage(null);
    try {
      const prediction = await predictScanVolume(file, token!);
      setResult({
        riskScore: prediction.risk_score,
        diagnosis: prediction.diagnosis,
        severity: severityFromRisk(prediction.risk_score),
        backend: prediction.backend,
        latencyMs: prediction.latency_ms,
        quantumTelemetry: quantumTelemetryFromPrediction(prediction),
        volume: volumeViewFromPrediction(prediction),
        provenance: provenanceFromPrediction(prediction),
      });
      setStatus("idle");
    } catch (err) {
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
    setStatus("idle");
    setErrorMessage(null);
    setResult(mockDiagnosticResult(activeCase));
  }

  const visibleResult = status === "error" ? null : result;

  return (
    <div className="flex flex-1 flex-col">
      <StudyHeader
        caseLabel={activeCase.label}
        result={visibleResult}
        apiHealth={apiHealth}
        onOpenCases={() => setCasesOpen(true)}
      />

      <div className="grid flex-1 grid-cols-1 md:grid-cols-[minmax(0,1fr)_360px] lg:grid-cols-[260px_minmax(0,1fr)_380px]">
        <aside className="hidden lg:block lg:border-r lg:border-surface-3" aria-label="Case navigation">
          <CaseNav
            activeCaseId={activeCase.id}
            onSelectCase={handleSelectCase}
            canDiagnose={Boolean(canDiagnose)}
            apiHealth={apiHealth}
            onUpload={handleUpload}
          />
        </aside>

        <Drawer open={casesOpen} onClose={() => setCasesOpen(false)} title="Demo Cases & Upload">
          <CaseNav
            activeCaseId={activeCase.id}
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

      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)} />
    </div>
  );
}
