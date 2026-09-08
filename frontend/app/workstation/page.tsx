"use client";

import { useRef, useState } from "react";
import { StudyHeader } from "@/components/workstation/StudyHeader";
import { StudySelector } from "@/components/workstation/StudySelector";
import { MRIViewer } from "@/components/workstation/MRIViewer";
import { PredictionPanel } from "@/components/workstation/PredictionPanel";
import { QuantumTelemetry } from "@/components/workstation/QuantumTelemetry";
import { ExplanationPanel } from "@/components/workstation/ExplanationPanel";
import { TechnicalDetails } from "@/components/workstation/TechnicalDetails";
import { ReportExport } from "@/components/workstation/ReportExport";
import { LoadingState } from "@/components/shared/LoadingState";
import { ErrorState } from "@/components/shared/ErrorState";
import { EmptyState } from "@/components/shared/EmptyState";
import { AuthModal } from "@/components/auth/AuthModal";
import { useAuth } from "@/lib/auth-context";
import { ApiError, predictScanVolume } from "@/lib/api";
import { PRESET_CASES, mockDiagnosticResult, severityFromRisk } from "@/lib/mock-data";
import { quantumTelemetryFromPrediction } from "@/lib/quantum-telemetry";
import { provenanceFromPrediction } from "@/lib/provenance";
import { volumeViewFromPrediction } from "@/lib/viewer";
import type { DiagnosticResult, PresetCase } from "@/lib/types";

type Status = "idle" | "loading" | "error";

export default function WorkstationPage() {
  const { token, user, isReady } = useAuth();
  const [activeCase, setActiveCase] = useState<PresetCase>(PRESET_CASES[0]);
  const [result, setResult] = useState<DiagnosticResult | null>(() => mockDiagnosticResult(PRESET_CASES[0]));
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [authOpen, setAuthOpen] = useState(false);
  const lastFileRef = useRef<File | null>(null);

  const canDiagnose = isReady && user?.role === "radiologist";

  function handleSelectCase(preset: PresetCase) {
    setActiveCase(preset);
    setStatus("idle");
    setErrorMessage(null);
    setResult(mockDiagnosticResult(preset));
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

  return (
    <div className="flex flex-1 flex-col">
      <StudyHeader caseLabel={activeCase.label} result={status === "error" ? null : result} onUpload={handleUpload} />

      <div className="grid flex-1 grid-cols-1 lg:grid-cols-[220px_1.6fr_1fr]">
        <aside className="border-b border-surface-3 lg:border-b-0 lg:border-r">
          <StudySelector activeCaseId={activeCase.id} onSelectCase={handleSelectCase} canDiagnose={Boolean(canDiagnose)} />
        </aside>

        <section className="min-h-[420px] border-b border-surface-3 lg:border-b-0 lg:border-r" aria-label="MRI viewer">
          <MRIViewer result={status === "error" ? null : result} />
        </section>

        <section className="flex flex-col gap-6 p-4 sm:p-6" aria-label="AI analysis">
          {status === "loading" && <LoadingState />}

          {status === "error" && (
            <ErrorState message={errorMessage ?? "Unknown error."} onRetry={handleRetry} onLoadDemo={handleLoadDemo} />
          )}

          {status === "idle" && result && (
            <>
              <PredictionPanel result={result} />
              <QuantumTelemetry telemetry={result.quantumTelemetry} provenance={result.provenance} />
              <ExplanationPanel volume={result.volume} />
              <TechnicalDetails result={result} />
              <ReportExport result={result} caseLabel={activeCase.label} />
            </>
          )}

          {status === "idle" && !result && (
            <EmptyState title="No result yet" description="Select a demo case or upload a scan to begin." />
          )}
        </section>
      </div>

      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)} />
    </div>
  );
}
