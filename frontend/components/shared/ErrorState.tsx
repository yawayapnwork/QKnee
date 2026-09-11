import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/Button";

/**
 * The honest response to a failed live inference call: state the failure
 * plainly, offer a retry, offer a clearly-labeled demo case -- never a
 * fabricated successful result. Callers must not render `PredictionPanel`
 * at all when this is shown; there is no "successful-looking" fallback UI.
 */
export function ErrorState({
  title = "Analysis unavailable",
  message,
  onRetry,
  onLoadDemo,
  actionLabel,
  onAction,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
  onLoadDemo?: () => void;
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div role="alert" className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      <AlertTriangle className="h-5 w-5 text-danger" aria-hidden="true" />
      <p className="text-sm font-medium text-ink-primary">{title}</p>
      <p className="max-w-sm text-xs text-ink-muted">{message}</p>
      <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
        {onAction && actionLabel && (
          <Button variant="primary" size="sm" onClick={onAction}>
            {actionLabel}
          </Button>
        )}
        {onRetry && (
          <Button variant="secondary" size="sm" onClick={onRetry}>
            Retry
          </Button>
        )}
        {onLoadDemo && (
          <Button variant="tertiary" size="sm" onClick={onLoadDemo}>
            Load Demo Case
          </Button>
        )}
      </div>
    </div>
  );
}
