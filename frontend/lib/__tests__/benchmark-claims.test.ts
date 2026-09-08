import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const HERO = join(__dirname, "..", "..", "components", "landing", "Hero.tsx");
const BENCHMARKS_TABLE = join(__dirname, "..", "..", "components", "landing", "BenchmarksTable.tsx");

describe("AUDIT.md C4a / architecture#10 regression: landing page claims match RESULTS.md", () => {
  it("never claims a Stanford MRNet validation cohort -- this project only evaluated on real RSNA Knee data", () => {
    const source = readFileSync(HERO, "utf-8");
    expect(source).not.toMatch(/MRNet/i);
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
});
