import { Skeleton } from "@/components/ui/Skeleton";

/** Shape-matched skeleton for the prediction panel while a live request is
 * in flight -- not a centered spinner, so the layout doesn't jump when the
 * real result arrives. */
export function LoadingState() {
  return (
    <div className="flex flex-col items-center gap-4 px-4 py-8" aria-busy="true" aria-live="polite">
      <span className="sr-only">Running inference</span>
      <Skeleton className="h-32 w-32 rounded-full sm:h-36 sm:w-36" />
      <Skeleton className="h-4 w-24" />
      <div className="grid w-full grid-cols-2 gap-3">
        <Skeleton className="h-12" />
        <Skeleton className="h-12" />
      </div>
    </div>
  );
}
