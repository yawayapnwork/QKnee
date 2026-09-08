import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const WORKSTATION_PAGE = join(__dirname, "..", "..", "app", "workstation", "page.tsx");

describe("AUDIT.md P0 #2 regression: preset telemetry cannot leak into live inference", () => {
  it("the live-upload handler never assigns a preset case's qubit values to a live result", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");

    // The exact fix for A3/P0#2 was deleting this assignment. Fail loudly if
    // it's ever reintroduced, however it's spelled.
    expect(source).not.toMatch(/qubitExpectations\s*:\s*activeCase\.qubitExpectations/);
    expect(source).not.toMatch(/activeCase\.qubitExpectations/);

    // The live-upload result must be built via quantumTelemetryFromPrediction
    // (the only function permitted to source live telemetry -- it takes no
    // preset/activeCase argument at all, see quantum-telemetry.ts).
    expect(source).toMatch(/quantumTelemetry:\s*quantumTelemetryFromPrediction\(prediction\)/);
  });
});
