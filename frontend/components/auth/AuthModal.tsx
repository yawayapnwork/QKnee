"use client";

import { useState } from "react";
import { X, Loader2, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { ApiError, loginClinician, registerClinician } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import type { Role } from "@/lib/types";
import { cn } from "@/lib/utils";

export function AuthModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { signIn } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [role, setRole] = useState<Role>("radiologist");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result =
        mode === "login"
          ? await loginClinician({ username: email, password })
          : await registerClinician({ email, password, full_name: fullName, role });
      signIn(result.access_token, result.user);
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Unable to reach the Q-Knee API. Try again shortly.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-2xl border border-slate-800 bg-slate-900 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-teal-400" />
            <h2 className="font-semibold text-slate-100">Clinician Access</h2>
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-200">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="flex border-b border-slate-800 px-6 pt-4 gap-6 text-sm">
          {(["login", "register"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={cn(
                "pb-3 font-medium transition-colors",
                mode === m ? "border-b-2 border-teal-400 text-teal-400" : "text-slate-500 hover:text-slate-300",
              )}
            >
              {m === "login" ? "Sign In" : "Request Access"}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 px-6 py-5">
          {mode === "register" && (
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">Full Name</label>
              <input
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Dr. Jane Doe"
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-teal-500"
              />
            </div>
          )}

          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">Institutional Email</label>
            <input
              required
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="clinician@hospital.org"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-teal-500"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-slate-400">Password</label>
            <input
              required
              type="password"
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-teal-500"
            />
          </div>

          {mode === "register" && (
            <div>
              <label className="mb-1 block text-xs font-medium text-slate-400">Requested Role</label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value as Role)}
                className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none focus:border-teal-500"
              >
                <option value="radiologist">Radiologist (full diagnostic access)</option>
                <option value="researcher">Researcher (view-only)</option>
                <option value="clinical_auditor">Clinical Auditor (view-only)</option>
              </select>
            </div>
          )}

          {error && (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-400">
              {error}
            </div>
          )}

          <Button type="submit" disabled={loading} className="w-full">
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {mode === "login" ? "Sign In" : "Create Account"}
          </Button>

          <p className="text-center text-[11px] leading-relaxed text-slate-500">
            Investigational research prototype. Not for clinical use — findings require independent
            radiologist over-read.
          </p>
        </form>
      </div>
    </div>
  );
}
