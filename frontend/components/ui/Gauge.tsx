import { cn } from "@/lib/utils";

const TONES: Record<string, string> = {
  Normal: "#10b981",
  Indeterminate: "#f59e0b",
  "Urgent Surgical Consult": "#f43f5e",
};

export function Gauge({ value, severity }: { value: number; severity: string }) {
  const clamped = Math.min(1, Math.max(0, value));
  const circumference = 2 * Math.PI * 54;
  const offset = circumference * (1 - clamped);
  const color = TONES[severity] ?? "#14b8a6";

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative flex h-36 w-36 items-center justify-center">
        <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
          <circle cx="60" cy="60" r="54" fill="none" stroke="#1e293b" strokeWidth="10" />
          <circle
            cx="60"
            cy="60"
            r="54"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-all duration-700 ease-out"
          />
        </svg>
        <div className="absolute flex flex-col items-center">
          <span className="font-mono text-3xl font-bold text-slate-50">{(clamped * 100).toFixed(1)}%</span>
        </div>
      </div>
      <span
        className={cn("max-w-[10rem] text-center text-[11px] font-semibold uppercase tracking-wide")}
        style={{ color }}
      >
        {severity}
      </span>
    </div>
  );
}
