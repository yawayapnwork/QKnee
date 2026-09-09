import { Loader2 } from "lucide-react";

/**
 * The honest response to "what's happening right now": a single
 * indeterminate state, not a fabricated multi-stage progress checklist.
 * `POST /api/v1/predict` (see `lib/api.ts`) is one request/response call —
 * ingest, feature extraction, quantum inference, and explanation all run
 * server-side inside it, and the API reports no intermediate stage
 * boundaries back to the client. A UI that ticked off "Feature
 * extraction ✓ / Quantum inference ●" during this wait would be inventing
 * progress the backend never sent — exactly the kind of fabricated signal
 * this product's provenance model exists to prevent. If the backend ever
 * starts reporting real stage events, this is the one place to wire them
 * in; until then, this stays a single honest "running" state.
 */
export function LoadingState() {
  return (
    <div className="flex flex-col items-center gap-3 rounded-md border border-surface-3 px-4 py-10 text-center" aria-busy="true" aria-live="polite">
      <Loader2 className="h-6 w-6 animate-spin text-accent" aria-hidden="true" />
      <p className="text-sm font-semibold text-ink-primary">Running inference</p>
      <p className="max-w-[220px] text-2xs leading-relaxed text-ink-faint">
        Feature extraction, quantum inference, and explanation all run server-side in one request — no intermediate
        progress is reported.
      </p>
    </div>
  );
}
