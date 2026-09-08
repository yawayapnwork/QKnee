export function MethodsSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-t border-surface-3 py-8 first:border-t-0 first:pt-0">
      <h2 className="text-lg font-semibold text-ink-primary">{title}</h2>
      <div className="mt-3 space-y-3 text-sm leading-relaxed text-ink-muted">{children}</div>
    </section>
  );
}
