"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import { Surface } from "@/components/ui/Surface";
import { Skeleton } from "@/components/ui/Skeleton";

const LABELS: Record<string, string> = {
  primary: "Primary (API unified head)",
  acl: "ACL",
  meniscus: "Meniscus",
  mcl: "MCL",
};

/**
 * Surfaces `GET /health`'s `model_status` field -- present on the backend
 * (`extras/api/server.py`) but, until this component, invisible anywhere
 * in the Next.js frontend (a real schema gap: the TS `HealthResponse` type
 * didn't even declare the field). This is the same checkpoint-availability
 * question the Streamlit dashboard's "Model Health" sidebar answers --
 * shown here so a reviewer of either interface sees the same facts.
 */
export function ModelStatusPanel() {
  const [status, setStatus] = useState<Record<string, string> | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetchHealth(controller.signal)
      .then((health) => setStatus(health.model_status ?? {}))
      .catch(() => setFailed(true));
    return () => controller.abort();
  }, []);

  return (
    <Surface className="p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Model checkpoint status</h3>
      <p className="mt-1 text-xs text-ink-muted">
        Live from the API&apos;s <code className="text-ink-primary">GET /health</code>. &ldquo;Unavailable&rdquo; means
        no trained checkpoint exists for that head — never presented as a working prediction.
      </p>

      <dl className="mt-3 space-y-1.5 font-mono text-xs">
        {failed && <p className="text-ink-faint">API unreachable — status not available.</p>}
        {!failed && !status && (
          <>
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
          </>
        )}
        {status &&
          Object.entries(status).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between">
              <dt className="text-ink-muted">{LABELS[key] ?? key}</dt>
              <dd className={value === "available" ? "text-status-live" : "text-ink-faint"}>
                {value === "available" ? "Available" : "Unavailable"}
              </dd>
            </div>
          ))}
      </dl>
    </Surface>
  );
}
