import { Hero } from "@/components/landing/Hero";
import { ArchitectureDiagram } from "@/components/landing/ArchitectureDiagram";

export default function HomePage() {
  return (
    <>
      <Hero />
      <ArchitectureDiagram />
      <footer className="border-t border-surface-3 px-4 py-8 text-center text-xs text-ink-faint sm:px-6">
        Q-Knee — investigational research prototype, not for clinical use.{" "}
        <a href="/methods" className="text-accent hover:underline">
          Full methods & evaluation
        </a>
        .
      </footer>
    </>
  );
}
