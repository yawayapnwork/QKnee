"use client";

import { useState } from "react";
import { CommandBar } from "@/components/workstation/CommandBar";
import { MriViewport } from "@/components/workstation/MriViewport";
import { TriageCard } from "@/components/workstation/TriageCard";
import { AuthModal } from "@/components/auth/AuthModal";
import { useAuth } from "@/lib/auth-context";
import { ApiError, predictScanVolume } from "@/lib/api";
import { PRESET_CASES, mockDiagnosticResult, severityFromRisk } from "@/lib/mock-data";
import type { DiagnosticResult, PresetCase } from "@/lib/types";

export default function WorkstationPage() {
  const { token, user, isReady } = useAuth();
  const [activeCase, setActiveCase] = useState<PresetCase>(PRESET_CASES[0]);
  const [result, setResult] = useState<DiagnosticResult | null>(() => mockDiagnosticResult(PRESET_CASES[0]));
  const [loading, setLoading] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canDiagnose = isReady && user?.role === "radiologist";

  function handleSelectCase(preset: PresetCase) {
    setActiveCase(preset);
    setError(null);
    setResult(mockDiagnosticResult(preset));
  }

  async function handleUpload(file: File) {
    setError(null);
    if (!token || !canDiagnose) {
      setAuthOpen(true);
      return;
    }
    setLoading(true);
    try {
      const prediction = await predictScanVolume(file, token);
      setResult({
        riskScore: prediction.risk_score,
        diagnosis: prediction.diagnosis,
        severity: severityFromRisk(prediction.risk_score),
        heatmap: prediction.gradcam_heatmap,
        backend: prediction.backend,
        latencyMs: prediction.latency_ms,
        // The live /predict response doesn't expose per-qubit expectations —
        // reuse the active preset's illustrative values so the telemetry
        // panel still renders something coherent alongside the real risk score.
        qubitExpectations: activeCase.qubitExpectations,
        source: "live",
      });
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.detail
          : "Backend unreachable (likely a Render cold start) — showing preset data instead.",
      );
      setResult(mockDiagnosticResult(activeCase));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-orthoc-bg">
      <CommandBar activeCaseId={activeCase.id} onSelectCase={handleSelectCase} onOpenAuth={() => setAuthOpen(true)} />

      <div className="mx-auto grid max-w-7xl grid-cols-1 gap-6 px-6 py-6 lg:grid-cols-[1.3fr_1fr]">
        <MriViewport result={result} onUpload={handleUpload} />
        <TriageCard result={result} loading={loading} caseLabel={activeCase.label} canDiagnose={Boolean(canDiagnose)} />
      </div>

      {error && (
        <div className="mx-auto max-w-7xl px-6 pb-6">
          <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-400">
            {error}
          </div>
        </div>
      )}

      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)} />
    </main>
  );
}
