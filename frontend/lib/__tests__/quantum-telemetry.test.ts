import { describe, expect, it } from "vitest";
import { quantumTelemetryFromPrediction } from "../quantum-telemetry";
import { provenanceFromPrediction } from "../provenance";
import { livePrediction } from "./fixtures";

// This file used to test a SECOND, independent provenance vocabulary
// (`QuantumTelemetry.provenance: "live" | "precomputed-demo" | "unavailable"`)
// that duplicated `lib/provenance.ts`'s `Provenance` enum with different
// spelling. That field is deleted (execution mandate rules 15/16/18: one
// authoritative provenance model, no competing vocabulary) -- telemetry
// availability/trust is now read from `provenanceFromPrediction(...)
// .quantumExecution` exclusively. These tests check the raw-number
// extraction only.

describe("quantumTelemetryFromPrediction", () => {
  it("has no parameter through which preset/activeCase data could reach it", () => {
    expect(quantumTelemetryFromPrediction.length).toBe(1);
  });

  it("extracts the real per-qubit values, qubit count, and device from a live response", () => {
    const telemetry = quantumTelemetryFromPrediction(livePrediction());
    expect(telemetry.expectations).toEqual([-0.31, 0.05, 0.77, -0.62]);
    expect(telemetry.nQubits).toBe(4);
    expect(telemetry.device).toBe("default.qubit");
  });

  it("extracts the same real values for a real, offline-scored demo case (GET /api/cases/{id}) -- no separate preset code path", () => {
    // scripts/build_real_demo_cases.py tags real demo cases
    // backend="cache-fallback/<study_uid>", but the wire shape (and this
    // function's job) is identical to a live /predict response.
    const prediction = livePrediction({
      backend: "cache-fallback/1.2.826.0.1.3680043.8.498.example",
      quantum_expectations: [0.12, -0.34, 0.56, -0.78],
    });
    const telemetry = quantumTelemetryFromPrediction(prediction);
    expect(telemetry.expectations).toEqual([0.12, -0.34, 0.56, -0.78]);
  });

  it("this component does not decide trust/availability -- that is provenanceFromPrediction's job", () => {
    const prediction = livePrediction({ backend: "cache-fallback/case_0001", quantum_expectations: [0.1, 0.2, 0.3, 0.4] });
    const telemetry = quantumTelemetryFromPrediction(prediction);
    const provenance = provenanceFromPrediction(prediction);

    expect(telemetry.expectations).toEqual([0.1, 0.2, 0.3, 0.4]);
    // The single source of truth for "is this trustworthy/available" is
    // the shared provenance object, read directly from the API's own
    // `quantum_execution` field -- never re-derived here.
    expect(provenance.quantumExecution).toBe(prediction.quantum_execution);
  });

  it("never fabricates values: reports empty expectations when the backend sent none", () => {
    const telemetry = quantumTelemetryFromPrediction(
      livePrediction({ backend: "mock", quantum_expectations: null, n_qubits: null, quantum_backend: null }),
    );
    expect(telemetry.expectations).toEqual([]);
    expect(telemetry.nQubits).toBeNull();
    expect(telemetry.device).toBeNull();
  });

  it("treats an empty quantum_expectations array the same as null", () => {
    const telemetry = quantumTelemetryFromPrediction(livePrediction({ quantum_expectations: [] }));
    expect(telemetry.expectations).toEqual([]);
  });

  it("does not derive expectations from risk_score: two responses with identical risk_score but different quantum_expectations stay distinct", () => {
    const a = quantumTelemetryFromPrediction(
      livePrediction({ risk_score: 0.5, quantum_expectations: [0.9, 0.9, 0.9, 0.9] }),
    );
    const b = quantumTelemetryFromPrediction(
      livePrediction({ risk_score: 0.5, quantum_expectations: [-0.9, -0.9, -0.9, -0.9] }),
    );
    expect(a.expectations).not.toEqual(b.expectations);
  });
});

describe("no client-side constructor for demo-case telemetry", () => {
  it("quantumTelemetryFromPreset is deleted along with the hand-authored preset data it served", async () => {
    const telemetryModule: Record<string, unknown> = await import("../quantum-telemetry");
    expect(telemetryModule.quantumTelemetryFromPreset).toBeUndefined();
  });
});
