"use client";

import { useState } from "react";
import Link from "next/link";
import { Activity, LogOut } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { AuthModal } from "@/components/auth/AuthModal";
import { useAuth } from "@/lib/auth-context";

export function Navbar() {
  const { user, signOut, isReady } = useAuth();
  const [authOpen, setAuthOpen] = useState(false);

  return (
    <>
      <header className="sticky top-0 z-40 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-teal-500 to-cyan-400">
              <Activity className="h-4.5 w-4.5 text-slate-950" strokeWidth={2.5} />
            </div>
            <span className="font-semibold tracking-tight text-slate-100">Q-Knee</span>
            <span className="hidden text-xs font-medium text-slate-500 sm:inline">Diagnostic Platform</span>
          </Link>

          <nav className="flex items-center gap-3">
            <Link
              href="/workstation"
              className="hidden text-sm font-medium text-slate-400 hover:text-slate-100 sm:inline"
            >
              Workstation
            </Link>
            {isReady && user ? (
              <div className="flex items-center gap-3">
                <span className="hidden text-xs text-slate-400 sm:inline">
                  {user.full_name} · <span className="text-teal-400">{user.role}</span>
                </span>
                <Button variant="ghost" size="sm" onClick={signOut}>
                  <LogOut className="h-3.5 w-3.5" />
                  Sign Out
                </Button>
              </div>
            ) : (
              <Button variant="secondary" size="sm" onClick={() => setAuthOpen(true)}>
                Clinician Sign In
              </Button>
            )}
          </nav>
        </div>
      </header>
      <AuthModal open={authOpen} onClose={() => setAuthOpen(false)} />
    </>
  );
}
