"use client";

import { useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { Modal } from "@/components/ui/Modal";
import { Button } from "@/components/ui/Button";
import { Field, SelectField } from "@/components/ui/Field";
import { Alert } from "@/components/ui/Alert";
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
  const firstFieldRef = useRef<HTMLInputElement>(null);

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
    <Modal open={open} onClose={onClose} titleId="auth-modal-title" title="Clinician Access">
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

        {error && <Alert tone="error">{error}</Alert>}

        <Button type="submit" disabled={loading} className="w-full">
          {loading && <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />}
          {mode === "login" ? "Sign In" : "Create Account"}
        </Button>

        <p className="text-center text-2xs leading-relaxed text-ink-faint">
          Investigational research prototype. Not for clinical use — findings require independent
          radiologist over-read.
        </p>
      </form>
    </Modal>
  );
}
