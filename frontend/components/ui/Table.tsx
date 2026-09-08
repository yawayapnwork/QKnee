import { cn } from "@/lib/utils";

/**
 * The one table primitive -- used by `BenchmarksTable` and
 * `ModelStatusPanel`. Dense by design (tight vertical padding, small
 * type): a data table in a clinical/research tool should read like a
 * spreadsheet, not a marketing feature-comparison grid.
 */
export function Table({ children, caption }: { children: React.ReactNode; caption: string }) {
  return (
    <div className="overflow-x-auto rounded-md border border-surface-3">
      <table className="w-full text-sm">
        <caption className="sr-only">{caption}</caption>
        {children}
      </table>
    </div>
  );
}

export function TableHead({ children }: { children: React.ReactNode }) {
  return (
    <thead>
      <tr className="border-b border-surface-3 bg-surface-2 text-left text-2xs uppercase tracking-wide text-ink-faint">
        {children}
      </tr>
    </thead>
  );
}

export function Th({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <th scope="col" className={cn("px-4 py-2.5 font-medium", className)}>
      {children}
    </th>
  );
}

export function Tr({ children, className }: { children: React.ReactNode; className?: string }) {
  return <tr className={cn("border-b border-surface-3 last:border-0", className)}>{children}</tr>;
}

export function Td({
  children,
  className,
  header,
}: {
  children: React.ReactNode;
  className?: string;
  header?: boolean;
}) {
  const Component = header ? "th" : "td";
  return (
    <Component
      scope={header ? "row" : undefined}
      className={cn("px-4 py-2.5", header ? "text-left font-medium text-ink-primary" : "text-ink-muted", className)}
    >
      {children}
    </Component>
  );
}
