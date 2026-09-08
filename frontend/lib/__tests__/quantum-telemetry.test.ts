import { describe, expect, it } from "vitest";
import { quantumTelemetryFromPrediction, quantumTelemetryFromPreset } from "../quantum-telemetry";
import { provenanceFromPrediction } from "../provenance";
import { PRESET_CASES } from "../mock-data";
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
  it("AUDIT.md P0 #2 regression: never returns a preset's own qubit values for a live response", () => {
    const prediction = livePrediction();
    const telemetry = quantumTelemetryFromPrediction(prediction);

    for (const preset of PRESET_CASES) {
      expect(telemetry.expectations).not.toEqual(preset.qubitExpectations);
    }
    expect(telemetry.expectations).toEqual(prediction.quantum_expectations);
  });

  it("has no parameter through which preset/activeCase data could reach it", () => {
    expect(quantumTelemetryFromPrediction.length).toBe(1);
  });

  it("extracts the real per-qubit values, qubit count, and device from a live response", () => {
    const telemetry = quantumTelemetryFromPrediction(livePrediction());
    expect(telemetry.expectations).toEqual([-0.31, 0.05, 0.77, -0.62]);
    expect(telemetry.nQubits).toBe(4);
    expect(telemetry.device).toBe("default.qubit");
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

describe("quantumTelemetryFromPreset", () => {
  it("returns the preset's real, precomputed per-qubit values", () => {
    for (const preset of PRESET_CASES) {
      const telemetry = quantumTelemetryFromPreset(preset);
      expect(telemetry.expectations).toEqual(preset.qubitExpectations);
      expect(telemetry.nQubits).toBe(preset.qubitExpectations.length);
    }
  });
});
