import Link from "next/link";
import { ArrowRight, BookOpen, Cpu } from "lucide-react";
import { Badge } from "@/components/ui/Badge";
import { API_BASE_URL } from "@/lib/api";

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-slate-800/80 bg-grid">
      <div
        className="pointer-events-none absolute inset-0 -z-10"
        style={{
          background:
            "radial-gradient(50% 40% at 50% 0%, rgba(20,184,166,0.16) 0%, rgba(2,6,23,0) 70%)",
        }}
      />
      <div className="mx-auto max-w-5xl px-6 py-24 text-center sm:py-32">
        <Badge tone="teal" dot className="mb-6">
          NISQ-Ready Hybrid ML
        </Badge>
        <h1 className="text-4xl font-bold tracking-tight text-slate-50 sm:text-6xl">
          Q-Knee <span className="text-gradient">Diagnostic Platform</span>
        </h1>
        <p className="mx-auto mt-6 max-w-2xl text-balance text-base leading-relaxed text-slate-400 sm:text-lg">
          Accelerating orthopedic knee MRI triage by coupling deep spatial feature extraction with a
          4-qubit variational quantum classifier.
        </p>
        <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Link
            href="/workstation"
            className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-teal-500 to-cyan-400 px-6 py-3.5 text-sm font-semibold text-slate-950 shadow-md shadow-teal-500/10 transition-all hover:shadow-glow"
          >
            <Cpu className="h-4 w-4" />
            Launch Diagnostic Workstation
            <ArrowRight className="h-4 w-4" />
          </Link>
          <a
            href={`${API_BASE_URL}/docs`}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-6 py-3.5 text-sm font-semibold text-slate-200 transition-colors hover:bg-slate-800"
          >
            <BookOpen className="h-4 w-4" />
            API Swagger Docs
          </a>
        </div>
        <p className="mx-auto mt-8 max-w-xl text-xs text-slate-600">
          Investigational research prototype · Stanford MRNet validation cohort · Confirmatory
          radiologist over-read required
        </p>
      </div>
    </section>
  );
}
