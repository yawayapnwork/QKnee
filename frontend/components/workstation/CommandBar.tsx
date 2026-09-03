"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, ChevronDown, Wifi, WifiOff, Loader2 } from "lucide-react";
import { fetchHealth } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { PRESET_CASES } from "@/lib/mock-data";
import type { PresetCase } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/lib/utils";

type HealthState = "checking" | "online" | "offline";

export function CommandBar({
  activeCaseId,
  onSelectCase,
  onOpenAuth,
}: {
  activeCaseId: string;
  onSelectCase: (preset: PresetCase) => void;
  onOpenAuth: () => void;
}) {
  const { user, isReady } = useAuth();
  const [health, setHealth] = useState<HealthState>("checking");

  useEffect(() => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    fetchHealth(controller.signal)
      .then(() => setHealth("online"))
      .catch(() => setHealth("offline"))
      .finally(() => clearTimeout(timeout));
    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, []);

  return (
    <div className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/95 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3 px-6 py-3">
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center gap-2">
            <Activity className="h-4.5 w-4.5 text-teal-400" />
            <span className="font-semibold text-slate-100">Q-Knee Workstation</span>
          </Link>

          <HealthIndicator state={health} />
        </div>

        <div className="flex items-center gap-3">
          <SampleSelector activeCaseId={activeCaseId} onSelectCase={onSelectCase} />

          {isReady && user ? (
            <Badge tone={user.role === "radiologist" ? "teal" : "amber"}>
              {user.full_name} · {user.role === "radiologist" ? "Radiologist" : "Research Observer — View Only"}
            </Badge>
          ) : (
            <button
              onClick={onOpenAuth}
              className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
            >
              Sign In to Diagnose
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function HealthIndicator({ state }: { state: HealthState }) {
  if (state === "checking") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-slate-500">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Pinging API…
      </span>
    );
  }
  if (state === "online") {
    return (
      <span className="flex items-center gap-1.5 text-xs text-emerald-400">
        <Wifi className="h-3.5 w-3.5" /> API Online
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-xs text-amber-400">
      <WifiOff className="h-3.5 w-3.5" /> Cold Start / Offline — using preset data
    </span>
  );
}

function SampleSelector({
  activeCaseId,
  onSelectCase,
}: {
  activeCaseId: string;
  onSelectCase: (preset: PresetCase) => void;
}) {
  const [open, setOpen] = useState(false);
  const active = PRESET_CASES.find((c) => c.id === activeCaseId) ?? PRESET_CASES[0];

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs font-medium text-slate-200 hover:bg-slate-800"
      >
        {active.label}: {active.description}
        <ChevronDown className="h-3.5 w-3.5 text-slate-500" />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="absolute right-0 z-20 mt-2 w-72 rounded-lg border border-slate-700 bg-slate-900 p-1.5 shadow-xl">
            {PRESET_CASES.map((preset) => (
              <button
                key={preset.id}
                onClick={() => {
                  onSelectCase(preset);
                  setOpen(false);
                }}
                className={cn(
                  "block w-full rounded-md px-3 py-2 text-left text-xs hover:bg-slate-800",
                  preset.id === activeCaseId ? "bg-slate-800 text-teal-400" : "text-slate-300",
                )}
              >
                <div className="font-medium">{preset.label}</div>
                <div className="text-slate-500">{preset.description}</div>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
