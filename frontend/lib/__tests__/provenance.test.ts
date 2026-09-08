import { describe, expect, it } from "vitest";
import { provenanceForFailedLiveFallback, provenanceForPreset, provenanceFromPrediction } from "../provenance";
import { mockDiagnosticResult } from "../mock-data";
import { PRESET_CASES } from "../mock-data";
import { livePrediction } from "./fixtures";

describe("AUDIT.md P1 #5/#7 regression: provenance survives backend -> API -> frontend", () => {
  it("provenanceFromPrediction passes the API's own provenance fields through unchanged", () => {
    const prediction = livePrediction({
      provenance: "live",
      provenance_label: "LIVE",
      model_source: "trained_checkpoint",
      model_source_label: "TRAINED MODEL",
      quantum_execution: "quantum_simulator",
      quantum_execution_label: "QUANTUM SIMULATOR",
    });

    const info = provenanceFromPrediction(prediction);

    expect(info.provenance).toBe("live");
    expect(info.provenanceLabel).toBe("LIVE");
    expect(info.modelSource).toBe("trained_checkpoint");
    expect(info.modelSourceLabel).toBe("TRAINED MODEL");
    expect(info.quantumExecution).toBe("quantum_simulator");
    expect(info.quantumExecutionLabel).toBe("QUANTUM SIMULATOR");
    expect(info.isTrustworthy).toBe(true);
  });

  it("AUDIT.md C4b regression: a 'live' backend tag with model_source 'random_fallback' still surfaces as mock_fallback provenance from the API, and the frontend must not override it back to trustworthy", () => {
    // The API (qknee.observability.provenance.classify) downgrades this
    // exact case server-side -- the frontend's only job is to not silently
    // re-promote it. This locks that contract in from the frontend side.
    const prediction = livePrediction({
      backend: "live",
      provenance: "mock_fallback",
      provenance_label: "MOCK/FALLBACK",
      model_source: "random_fallback",
      model_source_label: "MODEL FALLBACK (untrained weights)",
    });

    const info = provenanceFromPrediction(prediction);

    expect(info.provenance).toBe("mock_fallback");
    expect(info.isTrustworthy).toBe(false);
    expect(info.modelSource).toBe("random_fallback");
  });

  it("never fabricates a model_source when the API sent null", () => {
    const prediction = livePrediction({ model_source: null, model_source_label: null });
    const info = provenanceFromPrediction(prediction);
    expect(info.modelSource).toBeNull();
    expect(info.modelSourceLabel).toBeNull();
  });

  it("has no parameter through which preset/activeCase data could reach it", () => {
    // Structural guarantee mirroring quantum-telemetry.test.ts's pattern: a
    // live result's provenance is physically incapable of being sourced
    // from a preset case.
    expect(provenanceFromPrediction.length).toBe(1);
  });
});

describe("preset provenance is always explicitly marked, never disguised as live", () => {
  it("provenanceForPreset always returns 'precomputed_demo'", () => {
    const info = provenanceForPreset();
    expect(info.provenance).toBe("precomputed_demo");
    expect(info.provenanceLabel).toBe("PRECOMPUTED DEMO");
    expect(info.isTrustworthy).toBe(true);
  });

  it("selecting any preset case yields precomputed_demo provenance, not live", () => {
    for (const preset of PRESET_CASES) {
      const result = mockDiagnosticResult(preset);
      expect(result.provenance.provenance).toBe("precomputed_demo");
    }
  });
});

describe("AUDIT.md P1 #7 requirements 6/7: a failed live call never silently looks trustworthy", () => {
  it("provenanceForFailedLiveFallback is the loud mock_fallback state, distinct from a deliberately-selected preset", () => {
    const failed = provenanceForFailedLiveFallback();
    const selected = provenanceForPreset();

    expect(failed.provenance).toBe("mock_fallback");
    expect(failed.isTrustworthy).toBe(false);
    expect(failed.provenance).not.toBe(selected.provenance);
  });

  it("mockDiagnosticResult's isFailedLiveFallback flag switches provenance from precomputed_demo to mock_fallback for the exact same preset data", () => {
    const preset = PRESET_CASES[0];
    const selected = mockDiagnosticResult(preset, false);
    const failedFallback = mockDiagnosticResult(preset, true);

    expect(selected.riskScore).toBe(failedFallback.riskScore); // same underlying case
    expect(selected.provenance.provenance).toBe("precomputed_demo");
    expect(failedFallback.provenance.provenance).toBe("mock_fallback");
    expect(failedFallback.provenance.isTrustworthy).toBe(false);
  });
});
