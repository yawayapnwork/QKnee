import { describe, expect, it } from "vitest";
import { quantumTelemetryFromPrediction, quantumTelemetryFromPreset } from "../quantum-telemetry";
import { PRESET_CASES } from "../mock-data";
import { livePrediction } from "./fixtures";

describe("quantumTelemetryFromPrediction", () => {
  it("AUDIT.md P0 #2 regression: never returns a preset's own qubit values for a live response", () => {
    // Every preset's expectations differ from the "live" response used here.
    // If quantumTelemetryFromPrediction ever started falling back to (or
    // being called with) preset/activeCase data, this would start matching
    // one of them.
    const prediction = livePrediction();
    const telemetry = quantumTelemetryFromPrediction(prediction);

    for (const preset of PRESET_CASES) {
      expect(telemetry.expectations).not.toEqual(preset.qubitExpectations);
    }
    expect(telemetry.expectations).toEqual(prediction.quantum_expectations);
  });

  it("has no parameter through which preset/activeCase data could reach it", () => {
    // Structural guarantee, not just a value check: the function's only
    // input is the backend PredictionResponse, so a live result's telemetry
    // is physically incapable of being sourced from a preset case.
    expect(quantumTelemetryFromPrediction.length).toBe(1);
  });

  it("labels a genuine live backend response as provenance 'live'", () => {
    const telemetry = quantumTelemetryFromPrediction(livePrediction());
    expect(telemetry.provenance).toBe("live");
    expect(telemetry.expectations).toEqual([-0.31, 0.05, 0.77, -0.62]);
    expect(telemetry.nQubits).toBe(4);
    expect(telemetry.device).toBe("default.qubit");
  });

  it("labels a cache-fallback backend response as provenance 'precomputed-demo'", () => {
    const telemetry = quantumTelemetryFromPrediction(
      livePrediction({ backend: "cache-fallback/case_0001", quantum_expectations: [0.1, 0.2, 0.3, 0.4] }),
    );
    expect(telemetry.provenance).toBe("precomputed-demo");
    expect(telemetry.expectations).toEqual([0.1, 0.2, 0.3, 0.4]);
  });

  it("never fabricates values: reports 'unavailable' (empty expectations) when the backend sent none", () => {
    const telemetry = quantumTelemetryFromPrediction(
      livePrediction({ backend: "mock", quantum_expectations: null, n_qubits: null, quantum_backend: null }),
    );
    expect(telemetry.provenance).toBe("unavailable");
    expect(telemetry.expectations).toEqual([]);
  });

  it("treats an empty quantum_expectations array the same as null (unavailable, not an empty-but-'live' claim)", () => {
    const telemetry = quantumTelemetryFromPrediction(livePrediction({ quantum_expectations: [] }));
    expect(telemetry.provenance).toBe("unavailable");
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
  it("always labels preset/demo telemetry as 'precomputed-demo', never 'live'", () => {
    for (const preset of PRESET_CASES) {
      const telemetry = quantumTelemetryFromPreset(preset);
      expect(telemetry.provenance).toBe("precomputed-demo");
      expect(telemetry.expectations).toEqual(preset.qubitExpectations);
    }
  });
});
