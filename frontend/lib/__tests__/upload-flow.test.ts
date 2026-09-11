import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const WORKSTATION_PAGE = join(__dirname, "..", "..", "app", "workstation", "page.tsx");
const CASE_NAV = join(__dirname, "..", "..", "components", "workstation", "CaseNav.tsx");
const API_FILE = join(__dirname, "..", "api.ts");
const ERROR_STATE = join(__dirname, "..", "..", "components", "shared", "ErrorState.tsx");

describe("Q-Knee Workstation Upload Flow & Error Differentiation", () => {
  it("WorkstationPage defines the 5 distinct error categories", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/"initialization"/);
    expect(source).toMatch(/"credentials"/);
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

  it("WorkstationPage retains pending file when unauthenticated and auto-resumes on sign in", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/pendingFileRef\.current\s*=\s*file/);
    expect(source).toMatch(/token\s*&&\s*canDiagnose\s*&&\s*pendingFileRef\.current/);
  });

  it("WorkstationPage clears stale token with signOut() upon HTTP 401", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/err\.status === 401[\s\S]*?signOut\(\)/);
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
    expect(source).toMatch(/Bearer \$\{token\}/);
  });

  it("api.ts predictScanVolume sends multipart FormData with the file and conditional Bearer authorization", () => {
    const source = readFileSync(API_FILE, "utf-8");
    expect(source).toMatch(/formData\.append\("file",\s*file\)/);
    expect(source).toMatch(/Authorization:\s*`Bearer \$\{token\}`/);
    expect(source).toMatch(/token\?: string \| null/);
  });
});

describe("Q-Knee Dual-Mode Entry & Guest Access Flow", () => {
  const AUTH_CONTEXT = join(__dirname, "..", "auth-context.tsx");
  const HERO_FILE = join(__dirname, "..", "..", "components", "landing", "Hero.tsx");
  const FINAL_CTA = join(__dirname, "..", "..", "components", "landing", "FinalCta.tsx");
  const APP_SHELL = join(__dirname, "..", "..", "components", "layout", "AppShell.tsx");

  it("auth-context.tsx defines AuthMode and exports continueAsGuest", () => {
    const source = readFileSync(AUTH_CONTEXT, "utf-8");
    expect(source).toMatch(/export type AuthMode\s*=\s*"guest"\s*\|\s*"authenticated"/);
    expect(source).toMatch(/continueAsGuest:\s*\(\)\s*=>\s*void/);
    expect(source).toMatch(/authMode:\s*AuthMode/);
    expect(source).toMatch(/setAuthMode\("guest"\)/);
    expect(source).toMatch(/setAuthMode\("authenticated"\)/);
  });

  it("Hero.tsx renders both Continue as Guest and Sign In entry options", () => {
    const source = readFileSync(HERO_FILE, "utf-8");
    expect(source).toMatch(/Continue as Guest/);
    expect(source).toMatch(/Sign In/);
    expect(source).toMatch(/continueAsGuest/);
  });

  it("FinalCta.tsx renders both Continue as Guest and Sign In entry options", () => {
    const source = readFileSync(FINAL_CTA, "utf-8");
    expect(source).toMatch(/Continue as Guest/);
    expect(source).toMatch(/Sign In/);
    expect(source).toMatch(/continueAsGuest/);
  });

  it("AppShell.tsx displays Guest badge and Sign In button when unauthenticated", () => {
    const source = readFileSync(APP_SHELL, "utf-8");
    expect(source).toMatch(/authMode === "authenticated"/);
    expect(source).toMatch(/Guest/);
    expect(source).toMatch(/Sign In/);
    expect(source).toMatch(/Sign Out/);
  });

  it("CaseNav.tsx enables Live Analysis for guests and accepts isGuest prop", () => {
    const source = readFileSync(CASE_NAV, "utf-8");
    expect(source).toMatch(/isGuest\s*=\s*false/);
    expect(source).toMatch(/\(isGuest\s*\|\|\s*canDiagnose\)\s*&&\s*apiHealth === "online"/);
    expect(source).toMatch(/Live Analysis — Upload \.dcm \/ \.npy/);
  });

  it("WorkstationPage provides isGuest to CaseNav and does not block guest uploads", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/isGuest\s*=\s*authMode === "guest"/);
    expect(source).toMatch(/isGuest=\{isGuest\}/);
    expect(source).toMatch(/authMode === "authenticated" && !canDiagnose/);
  });
});
