import { Hero } from "@/components/landing/Hero";
import { EvidenceSection } from "@/components/landing/EvidenceSection";
import { ArchitectureSection } from "@/components/landing/ArchitectureSection";
import { ExplainabilitySection } from "@/components/landing/ExplainabilitySection";
import { QuantumSection } from "@/components/landing/QuantumSection";
import { FinalCta } from "@/components/landing/FinalCta";

export default function HomePage() {
  return (
    <>
      <Hero />
      <EvidenceSection />
      <ArchitectureSection />
      <ExplainabilitySection />
      <QuantumSection />
      <FinalCta />
      <footer className="border-t border-surface-3 px-4 py-8 text-center text-xs text-ink-faint sm:px-6">
        Research prototype — not a substitute for professional radiological diagnosis. Not a certified medical
        device.{" "}
        <a href="/methods" className="text-accent hover:underline">
          Full methods & evaluation
        </a>
        .
      </footer>
    </>
  );
}
