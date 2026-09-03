import { Navbar } from "@/components/landing/Navbar";
import { Hero } from "@/components/landing/Hero";
import { PipelineVisualizer } from "@/components/landing/PipelineVisualizer";
import { BenchmarksTable } from "@/components/landing/BenchmarksTable";

export default function HomePage() {
  return (
    <main>
      <Navbar />
      <Hero />
      <PipelineVisualizer />
      <BenchmarksTable />
      <footer className="border-t border-slate-800/80 px-6 py-10 text-center text-xs text-slate-600">
        Q-Knee Diagnostic Platform · AI &amp; Quantum Innovation Track · Investigational research
        prototype, not for clinical use.
      </footer>
    </main>
  );
}
