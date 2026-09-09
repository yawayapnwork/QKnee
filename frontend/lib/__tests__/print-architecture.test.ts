import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

/** Strips CSS/JS comments -- this test's job is to catch what actually
 * matches a selector or renders, not prose in a comment (including this
 * file's own explanatory comment about the bug it regression-tests). */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

const FRONTEND_ROOT = join(__dirname, "..", "..");
const GLOBALS_CSS = join(FRONTEND_ROOT, "app", "globals.css");
const PRINT_REPORT = join(FRONTEND_ROOT, "components", "workstation", "PrintReport.tsx");
const WORKSTATION_PAGE = join(FRONTEND_ROOT, "app", "workstation", "page.tsx");

// Every file that renders an INTERACTIVE control (a button, slider,
// toggle, upload input, or a full modal/drawer overlay) inside the
// workstation must carry the `no-print` class on its root, per the print
// architecture in `app/globals.css`. This is a source-level check, not a
// rendered-DOM one -- the project has no React-rendering test harness
// (jsdom/@testing-library/react) installed, and adding one for a single
// test would be a new dependency the remediation brief explicitly says to
// avoid unless necessary. This still catches the exact regression a
// hostile review found: a print stylesheet whose selectors don't actually
// match anything in the real DOM.
const FILES_THAT_MUST_CARRY_NO_PRINT = [
  join(FRONTEND_ROOT, "components", "workstation", "StudyHeader.tsx"),
  join(FRONTEND_ROOT, "components", "workstation", "StatusBar.tsx"),
  join(FRONTEND_ROOT, "components", "workstation", "ReportExport.tsx"),
  join(FRONTEND_ROOT, "components", "ui", "Drawer.tsx"),
  join(FRONTEND_ROOT, "components", "ui", "Modal.tsx"),
  join(FRONTEND_ROOT, "components", "workstation", "ExplanationWorkspace.tsx"),
  join(FRONTEND_ROOT, "components", "layout", "AppShell.tsx"),
];

describe("hostile-review regression: print output excludes interactive controls", () => {
  it("app/globals.css defines a class-based (not tag-based) print architecture", () => {
    const css = stripComments(readFileSync(GLOBALS_CSS, "utf-8"));
    expect(css).toMatch(/\.no-print\s*\{/);
    expect(css).toMatch(/\.print-only\s*\{/);
    expect(css).toMatch(/@media print/);
    // The specific defect: `header, aside { display: none }` matched
    // almost nothing in the real DOM, because the workstation's own
    // toolbar/status-bar/export buttons never lived inside those tags.
    expect(css).not.toMatch(/header,\s*aside(,\s*\.no-print)?\s*\{/);
  });

  it.each(FILES_THAT_MUST_CARRY_NO_PRINT)("%s marks its interactive root with `no-print`", (path) => {
    const source = readFileSync(path, "utf-8");
    expect(source).toMatch(/\bno-print\b/);
  });

  it("the main workstation grid (case nav, imaging viewer, AI analysis columns) is marked no-print", () => {
    const source = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(source).toMatch(/className="no-print grid/);
  });

  it("PrintReport renders no interactive controls -- no <button>, <input>, or <select>", () => {
    const source = readFileSync(PRINT_REPORT, "utf-8");
    expect(source).not.toMatch(/<button/i);
    expect(source).not.toMatch(/<input/i);
    expect(source).not.toMatch(/<select/i);
  });

  it("PrintReport is marked print-only and is rendered from the workstation page", () => {
    const reportSource = readFileSync(PRINT_REPORT, "utf-8");
    expect(reportSource).toMatch(/print-only/);
    const pageSource = readFileSync(WORKSTATION_PAGE, "utf-8");
    expect(pageSource).toMatch(/<PrintReport/);
  });

  it("PrintReport includes every required report field: identification, provenance, prediction, images, explanation, limitations, timestamp", () => {
    const source = readFileSync(PRINT_REPORT, "utf-8");
    expect(source).toMatch(/caseLabel/); // study/result identification
    expect(source).toMatch(/ProvenanceBadge/); // provenance
    expect(source).toMatch(/riskScore|formatPercent/); // prediction/risk result
    expect(source).toMatch(/baseImage/); // MRI image
    expect(source).toMatch(/gradcamImage/); // Grad-CAM image, when available
    expect(source).toMatch(/GRADCAM_DISCLAIMER/); // explanation
    expect(source).toMatch(/LIMITATIONS/); // limitations/disclaimer
    expect(source).toMatch(/toISOString/); // timestamp
  });
});
