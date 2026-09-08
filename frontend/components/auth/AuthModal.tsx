"use client";

import { useEffect, useRef, useState } from "react";
import { X, Loader2, ShieldCheck } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Field, SelectField } from "@/components/ui/Field";
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
  const [role, setRole] = useState<Role>("researcher");
  const [inviteCode, setInviteCode] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const firstFieldRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    firstFieldRef.current?.focus();
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [open, onClose]);

  if (!open) return null;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result =
        mode === "login"
          ? await loginClinician({ username: email, password })
          : await registerClinician({
              email,
              password,
              full_name: fullName,
              role,
              invite_code: role === "radiologist" ? inviteCode : undefined,
            });
      signIn(result.access_token, result.user);
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Unable to reach the Q-Knee API. Try again shortly.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-surface-0/85 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-modal-title"
        className="w-full max-w-md rounded-lg border border-surface-3 bg-surface-1 shadow-xl"
      >
        <div className="flex items-center justify-between border-b border-surface-3 px-6 py-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-accent" aria-hidden="true" />
            <h2 id="auth-modal-title" className="font-semibold text-ink-primary">
              Clinician Access
            </h2>
          </div>
          <button onClick={onClose} aria-label="Close dialog" className="text-ink-muted hover:text-ink-primary">
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <div role="tablist" aria-label="Authentication mode" className="flex gap-6 border-b border-surface-3 px-6 pt-4 text-sm">
          {(["login", "register"] as const).map((m) => (
            <button
              key={m}
              role="tab"
              aria-selected={mode === m}
              onClick={() => setMode(m)}
              className={cn(
                "pb-3 font-medium transition-colors",
                mode === m ? "border-b-2 border-accent text-accent" : "text-ink-muted hover:text-ink-primary",
              )}
            >
              {m === "login" ? "Sign In" : "Request Access"}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 px-6 py-5">
          {mode === "register" && (
            <Field label="Full Name" required value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Dr. Jane Doe" />
          )}

          <Field
            ref={firstFieldRef}
            label="Institutional Email"
            required
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="clinician@hospital.org"
          />

          <Field
            label="Password"
            required
            type="password"
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
          />

          {mode === "register" && (
            <SelectField label="Requested Role" value={role} onChange={(e) => setRole(e.target.value as Role)}>
              <option value="researcher">Researcher (view-only)</option>
              <option value="clinical_auditor">Clinical Auditor (view-only)</option>
              <option value="radiologist">Radiologist (full diagnostic access — requires invite code)</option>
            </SelectField>
          )}

          {mode === "register" && role === "radiologist" && (
            <Field
              label="Radiologist Invite Code"
              required
              value={inviteCode}
              onChange={(e) => setInviteCode(e.target.value)}
              placeholder="Provided by your institution"
            />
          )}

          {error && (
            <div role="alert" className="rounded-md border border-status-fallback/40 bg-status-fallback/10 px-3 py-2 text-xs text-status-fallback">
              {error}
            </div>
          )}

          <Button type="submit" disabled={loading} className="w-full">
            {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
            {mode === "login" ? "Sign In" : "Create Account"}
          </Button>

          <p className="text-center text-[11px] leading-relaxed text-ink-faint">
            Investigational research prototype. Not for clinical use — findings require independent
            radiologist over-read.
          </p>
        </form>
      </div>
    </div>
  );
}
