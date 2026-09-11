"use client";

import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import { Surface } from "@/components/ui/Surface";
import { Skeleton } from "@/components/ui/Skeleton";
import { Table, TableHead, Th, Tr, Td } from "@/components/ui/Table";

const LABELS: Record<string, string> = {
  primary: "Primary (API unified head)",
  acl: "ACL",
  meniscus: "Meniscus",
  mcl: "MCL",
};

/**
 * Surfaces `GET /health`'s `model_status` field -- present on the backend
 * (`extras/api/server.py`) but, until this component, invisible anywhere
 * in the Next.js frontend. This is the same checkpoint-availability
 * question the Streamlit dashboard's "Model Health" sidebar answers --
 * shown here so a reviewer of either interface sees the same facts.
 */
export function ModelStatusPanel() {
  const [status, setStatus] = useState<Record<string, string> | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let isMounted = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let abortCtrl: AbortController | null = null;

    async function loadStatus() {
      abortCtrl?.abort();
      const controller = new AbortController();
      abortCtrl = controller;
      try {
        const health = await fetchHealth(controller.signal);
        if (!isMounted) return;
        setStatus(health.model_status ?? {});
        setFailed(false);
      } catch {
        if (!isMounted) return;
        setFailed(true);
        // Retry after 5s while unreachable to automatically recover
        timer = setTimeout(() => void loadStatus(), 5000);
      }
    }

    void loadStatus();
    return () => {
      isMounted = false;
      if (timer) clearTimeout(timer);
      abortCtrl?.abort();
    };
  }, []);

  return (
    <Surface className="p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-faint">Model checkpoint status</h3>
      <p className="mt-1 text-xs text-ink-muted">
        Live from the API&apos;s <code className="text-ink-primary">GET /health</code>. &ldquo;Unavailable&rdquo; means
        no trained checkpoint exists for that head — never presented as a working prediction.
      </p>

      <div className="mt-3">
        {failed && <p className="text-xs text-ink-faint">API unreachable — status not available.</p>}
        {!failed && !status && (
          <div className="space-y-1.5">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-8 w-full" />
          </div>
        )}
        {status && (
          <Table caption="Model checkpoint availability by head">
            <TableHead>
              <Th>Head</Th>
              <Th>Status</Th>
            </TableHead>
            <tbody>
              {Object.entries(status).map(([key, value]) => (
                <Tr key={key}>
                  <Td header className="font-mono">
                    {LABELS[key] ?? key}
                  </Td>
                  <Td className={value === "available" ? "font-mono text-success" : "font-mono text-ink-faint"}>
                    {value === "available" ? "Available" : "Unavailable"}
                  </Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        )}
      </div>
    </Surface>
  );
}
