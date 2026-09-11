import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const WORKSTATION_PAGE = join(__dirname, "..", "..", "app", "workstation", "page.tsx");
const CASE_NAV = join(__dirname, "..", "..", "components", "workstation", "CaseNav.tsx");
const API_FILE = join(__dirname, "..", "api.ts");
const ERROR_STATE = join(__dirname, "..", "..", "components", "shared", "ErrorState.tsx");

describe("Q-Knee Workstation Upload Flow & Public Access", () => {
  it("WorkstationPage defines functional error categories", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/"initialization"/);
    expect(source).toMatch(/"file_selection"/);
    expect(source).toMatch(/"upload_request"/);
    expect(source).toMatch(/"inference"/);
  });

  it("WorkstationPage validates file extension for .dcm, .dicom, and .npy before upload", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/\.endsWith\("\.dcm"\)/);
    expect(source).toMatch(/\.endsWith\("\.dicom"\)/);
    expect(source).toMatch(/\.endsWith\("\.npy"\)/);
  });

  it("WorkstationPage checks for empty file (0 bytes)", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/file\.size === 0/);
  });

  it("WorkstationPage executes upload directly without requiring authentication", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).not.toMatch(/useAuth/);
    expect(source).not.toMatch(/canDiagnose/);
    expect(source).toMatch(/void runInference\(file\)/);
  });

  it("WorkstationPage handles upload errors gracefully without credential blockers or auth modals", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).not.toMatch(/signOut\(\)/);
    expect(source).not.toMatch(/AuthModal/);
  });

  it("CaseNav preserves cases: CaseSummary[] prop and upload input accept attribute", () => {
    const source = readFileSync(CASE_NAV, "utf-8");
    expect(source).toMatch(/cases:\s*CaseSummary\[\]/);
    expect(source).toMatch(/accept="\.dcm,\.dicom,\.npy"/);
    expect(source).toMatch(/isUploading/);
  });

  it("ErrorState supports title, actionLabel, and onAction", () => {
    const source = readFileSync(ERROR_STATE, "utf-8");
    expect(source).toMatch(/title\s*=\s*"Analysis unavailable"/);
    expect(source).toMatch(/actionLabel/);
    expect(source).toMatch(/onAction/);
  });

  it("api.ts exports fetchCurrentUser to query /api/v1/auth/me", () => {
    const source = readFileSync(API_FILE, "utf-8");
    expect(source).toMatch(/export async function fetchCurrentUser/);
    expect(source).toMatch(/\/api\/v1\/auth\/me/);
  });

  it("api.ts predictScanVolume sends multipart FormData with the file", () => {
    const source = readFileSync(API_FILE, "utf-8");
    expect(source).toMatch(/formData\.append\("file",\s*file\)/);
  });
});
