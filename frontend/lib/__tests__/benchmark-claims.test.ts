import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const FRONTEND_ROOT = join(__dirname, "..", "..");
const HERO = join(FRONTEND_ROOT, "components", "landing", "Hero.tsx");
const ARCHITECTURE_DIAGRAM = join(FRONTEND_ROOT, "components", "landing", "ArchitectureDiagram.tsx");
const BENCHMARKS_TABLE = join(FRONTEND_ROOT, "components", "methods", "BenchmarksTable.tsx");

function allSourceFiles(dir: string): string[] {
  const entries = readdirSync(dir, { withFileTypes: true });
  return entries.flatMap((entry) => {
    if (entry.name === "node_modules" || entry.name === ".next" || entry.name === "__tests__" || entry.name.startsWith("."))
      return [];
    const full = join(dir, entry.name);
    if (entry.isDirectory()) return allSourceFiles(full);
    if (/\.(tsx?|css)$/.test(entry.name)) return [full];
    return [];
  });
}

/** Strips comments before matching -- this test's job is to catch a claim
 * a VIEWER would see (JSX/rendered text), not a code comment explaining
 * which historical bug was fixed and quoting the old, now-deleted text. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

const ALL_SOURCE = [
  ...allSourceFiles(join(FRONTEND_ROOT, "app")),
  ...allSourceFiles(join(FRONTEND_ROOT, "components")),
  ...allSourceFiles(join(FRONTEND_ROOT, "lib")),
].map((path) => ({ path, content: stripComments(readFileSync(path, "utf-8")) }));

describe("AUDIT.md C4a / ARCHITECTURE.md / FRONTEND_REDESIGN.md regression: no fabricated or unsupported claims", () => {
  it("never claims a Stanford MRNet validation cohort -- this project only evaluated on real RSNA Knee data", () => {
    expect(readFileSync(HERO, "utf-8")).not.toMatch(/MRNet/i);
  });

  it("never reintroduces the fabricated 0.884/0.912 'Verified Clinical Benchmarks' numbers", () => {
    const source = readFileSync(BENCHMARKS_TABLE, "utf-8");
    expect(source).not.toMatch(/0\.884/);
    expect(source).not.toMatch(/0\.912/);
    expect(source).not.toMatch(/Verified Clinical Benchmarks/i);
  });

  it("benchmarks table cites its real source file (kaggle_benchmark_summary.json / RESULTS.md)", () => {
    const source = readFileSync(BENCHMARKS_TABLE, "utf-8");
    expect(source).toMatch(/kaggle_benchmark_summary\.json/);
    expect(source).toMatch(/RESULTS\.md/);
  });

  it("never claims 3D / volumetric / sagittal-coronal DICOM decoding anywhere in the frontend", () => {
    // Execution mandate's single most serious factual correction:
    // PipelineVisualizer.tsx used to state "Volumetric MRI Ingestion —
    // 3D sagittal / coronal DICOM & NumPy volume decoding," which the
    // backend contract does not support (extras/api/server.py marks
    // Coronal/Sagittal `available: false` for a single-series upload).
    const offenders = ALL_SOURCE.filter(({ content }) => /3D\s+(sagittal|coronal)|volumetric\s+(mri\s+)?ingestion/i.test(content));
    expect(offenders.map((o) => o.path)).toEqual([]);
  });

  it("never POSITIVELY claims quantum advantage, clinical superiority, or full volumetric reconstruction", () => {
    // A negated mention ("no quantum advantage is claimed") is the honest,
    // required disclosure (see RESULTS.md/Methods) -- only an
    // un-negated, affirmative claim is banned. Checked by requiring the
    // ~20 characters immediately before the match contain no negation.
    const bannedPhrases = [
      /quantum\s+advantage/gi,
      /clinically?\s+superior/gi,
      /full\s+volumetric\s+reconstruction/gi,
      /true\s+3D\s+mri\s+understanding/gi,
      /3D\s+quantum\s+mri/gi,
    ];
    const negationWords = /\b(no|not|without|never|isn't|doesn't|does not)\b/i;

    for (const phrase of bannedPhrases) {
      for (const { path, content } of ALL_SOURCE) {
        for (const match of content.matchAll(phrase)) {
          const precedingContext = content.slice(Math.max(0, (match.index ?? 0) - 24), match.index);
          if (!negationWords.test(precedingContext)) {
            throw new Error(`Unnegated banned phrase "${match[0]}" found in ${path}`);
          }
        }
      }
    }
  });

  it("never says 'SIMULATION MODE' or presents an ambiguous simulator-vs-fake label", () => {
    const offenders = ALL_SOURCE.filter(({ content }) => /SIMULATION MODE/i.test(content));
    expect(offenders.map((o) => o.path)).toEqual([]);
  });

  it("never labels a print-dialog trigger as 'Export PDF' -- window.print() does not generate a PDF file", () => {
    const offenders = ALL_SOURCE.filter(({ content }) => /export pdf/i.test(content));
    expect(offenders.map((o) => o.path)).toEqual([]);
  });

  it("has no second, competing provenance vocabulary (QuantumTelemetryProvenance, or a duplicated PROVENANCE_LABEL map)", () => {
    const offenders = ALL_SOURCE.filter(({ content }) => /QuantumTelemetryProvenance/.test(content));
    expect(offenders.map((o) => o.path)).toEqual([]);
  });

  it("architecture diagram describes only implemented stages, in the mandated wording", () => {
    const source = stripComments(readFileSync(ARCHITECTURE_DIAGRAM, "utf-8"));
    expect(source).toMatch(/Knee MRI/);
    expect(source).toMatch(/Visual Feature Extraction/);
    expect(source).toMatch(/Feature Compression/);
    expect(source).toMatch(/Variational Quantum Classifier/);
    expect(source).not.toMatch(/3D\s+(sagittal|coronal)/i);
  });
});
