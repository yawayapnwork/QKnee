import { describe, expect, it } from "vitest";
import { provenanceFromPrediction } from "../provenance";
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

  it("a real, offline-scored demo case (GET /api/cases/{id}) is marked precomputed_demo, exactly like any other cache-fallback response", () => {
    // Real demo cases are ordinary PredictionResponses with
    // backend === "cache-fallback/<study_uid>" (see
    // scripts/build_real_demo_cases.py) -- provenanceFromPrediction needs
    // no separate "preset" code path to classify them correctly, because
    // qknee.observability.provenance.classify already maps that backend
    // tag to "precomputed_demo" server-side.
    const prediction = livePrediction({
      backend: "cache-fallback/1.2.826.0.1.3680043.8.498.example",
      provenance: "precomputed_demo",
      provenance_label: "PRECOMPUTED DEMO",
      model_source: "trained_checkpoint",
      model_source_label: "TRAINED MODEL",
      quantum_execution: "quantum_simulator",
      quantum_execution_label: "QUANTUM SIMULATOR",
    });

    const info = provenanceFromPrediction(prediction);

    expect(info.provenance).toBe("precomputed_demo");
    expect(info.isTrustworthy).toBe(true);
    // Unlike the old hand-authored presets, a real demo case's quantum
    // telemetry IS backend-attested (a real circuit ran, once, offline) --
    // it earns the genuine "QUANTUM SIMULATOR" badge, not "not_verifiable".
    expect(info.quantumExecution).toBe("quantum_simulator");
  });
});

describe("execution mandate rule 13: a failed live call never silently becomes a mock prediction", () => {
  it("there is no client-side constructor for a 'failed live call' provenance -- mock_fallback can only ever come from the backend's own response", async () => {
    // `provenanceForFailedLiveFallback` used to exist specifically so a
    // failed fetch() could synthesize a DiagnosticResult labeled
    // mock_fallback. That function is deleted: app/workstation/page.tsx's
    // catch block renders `ErrorState` (no DiagnosticResult at all) instead
    // of calling into this module. The only way `provenance === "mock_fallback"`
    // can appear on screen is a genuine, successfully-received API payload
    // that says so itself (see the C4b regression test above).
    const provenanceModule: Record<string, unknown> = await import("../provenance");
    expect(provenanceModule.provenanceForFailedLiveFallback).toBeUndefined();
  });

  it("there is no client-side constructor for demo-case provenance either -- provenanceForPreset is deleted along with the hand-authored preset data it served", async () => {
    const provenanceModule: Record<string, unknown> = await import("../provenance");
    expect(provenanceModule.provenanceForPreset).toBeUndefined();
  });
});
