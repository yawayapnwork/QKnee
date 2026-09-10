import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const WORKSTATION_PAGE = join(__dirname, "..", "..", "app", "workstation", "page.tsx");
const CASE_NAV = join(__dirname, "..", "..", "components", "workstation", "CaseNav.tsx");

describe("AUDIT.md P0 #2 regression: preset telemetry cannot leak into live inference", () => {
  it("the live-upload handler never assigns a preset case's qubit values to a live result", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");

    // The exact fix for A3/P0#2 was deleting this assignment. Fail loudly if
    // it's ever reintroduced, however it's spelled.
    expect(source).not.toMatch(/qubitExpectations\s*:\s*activeCase\.qubitExpectations/);
    expect(source).not.toMatch(/activeCase\.qubitExpectations/);

    // Every DiagnosticResult this page builds -- live upload or demo case
    // alike -- goes through the one shared toDiagnosticResult conversion,
    // whose quantumTelemetry field is always quantumTelemetryFromPrediction
    // (the only function permitted to source telemetry -- it takes no
    // preset/activeCase argument at all, see quantum-telemetry.ts).
    expect(source).toMatch(/quantumTelemetry:\s*quantumTelemetryFromPrediction\(prediction\)/);
  });
});

describe("zero static preset cases bypass the API pipeline", () => {
  it("the workstation page never imports the old hand-authored preset data", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");

    // mock-data.ts (PRESET_CASES, mockDiagnosticResult) is deleted; fail
    // loudly if either it or a differently-named re-implementation of the
    // same "hand-authored fake case" concept is ever reintroduced here.
    expect(source).not.toMatch(/mock-data/);
    expect(source).not.toMatch(/PRESET_CASES/);
    expect(source).not.toMatch(/mockDiagnosticResult/);
  });

  it("demo cases are loaded exclusively through GET /api/cases and GET /api/cases/{id}", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");

    // fetchCases/fetchCase are the only two functions permitted to supply
    // this page's `cases` list / a selected case's result -- both go
    // through lib/api.ts's real HTTP calls to the FastAPI backend, never a
    // constant baked into the frontend bundle.
    expect(source).toMatch(/import\s*\{[^}]*\bfetchCases\b[^}]*\}\s*from\s*"@\/lib\/api"/);
    expect(source).toMatch(/import\s*\{[^}]*\bfetchCase\b[^}]*\}\s*from\s*"@\/lib\/api"/);
  });

  it("CaseNav renders only the cases its parent fetched from the API, never a bundled constant", () => {
    const source = readFileSync(CASE_NAV, "utf-8");

    expect(source).not.toMatch(/mock-data/);
    expect(source).not.toMatch(/PRESET_CASES/);
    // `cases` must be a prop (parent-supplied), not a module-level import.
    expect(source).toMatch(/cases:\s*CaseSummary\[\]/);
  });
});
