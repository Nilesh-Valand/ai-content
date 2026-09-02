import { Confidence } from "@/lib/api";
import { ShieldAlert, ShieldCheck, ShieldQuestion, ListChecks, FileText } from "lucide-react";

function verdictTheme(pct: number) {
  if (pct >= 60)
    return {
      label: "Likely AI-Written",
      ring: "#EF4444",
      ringSoft: "#FEE2E2",
      text: "text-danger-600",
      chip: "bg-danger-50 text-danger-600 border-danger-400/30",
      Icon: ShieldAlert,
    };
  if (pct >= 30)
    return {
      label: "Mixed Signals",
      ring: "#F59E0B",
      ringSoft: "#FEF3C7",
      text: "text-warn-600",
      chip: "bg-warn-50 text-warn-600 border-warn-400/30",
      Icon: ShieldQuestion,
    };
  return {
    label: "Likely Human-Written",
    ring: "#10B981",
    ringSoft: "#D1FAE5",
    text: "text-success-600",
    chip: "bg-success-50 text-success-600 border-success-400/30",
    Icon: ShieldCheck,
  };
}

export default function VerdictStamp({
  percentage,
  confidence,
  patternCount,
  sentenceCount,
}: {
  percentage: number;
  confidence: Confidence;
  patternCount: number;
  sentenceCount: number;
}) {
  const theme = verdictTheme(percentage);
  const Icon = theme.Icon;
  const angle = Math.min(Math.max(percentage, 0), 100) * 3.6;

  return (
    <div className="fade-in grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
      <div className="xl:col-span-2 rounded-2xl border border-border bg-surface shadow-card p-6 flex items-center gap-6">
        <div
          className="relative h-24 w-24 shrink-0 rounded-full grid place-items-center"
          style={{
            background: `conic-gradient(${theme.ring} ${angle}deg, ${theme.ringSoft} ${angle}deg)`,
          }}
        >
          <div className="h-[72px] w-[72px] rounded-full bg-surface grid place-items-center">
            <span className="text-xl font-bold text-ink">{percentage.toFixed(0)}%</span>
          </div>
        </div>
        <div className="min-w-0">
          <div
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${theme.chip}`}
          >
            <Icon className="h-3.5 w-3.5" strokeWidth={2.5} />
            {theme.label}
          </div>
          <p className="text-sm text-ink-muted mt-2 leading-snug">
            AI-writing likelihood from vocabulary, structure, and specificity signals —
            an estimate, not proof.
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-surface shadow-card p-6 flex flex-col justify-between">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-50 text-brand-600">
          <ListChecks className="h-[18px] w-[18px]" strokeWidth={2.25} />
        </span>
        <div className="mt-4">
          <p className="text-2xl font-bold text-ink">{confidence}</p>
          <p className="text-sm text-ink-muted mt-0.5">Confidence level</p>
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-surface shadow-card p-6 flex flex-col justify-between">
        <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-accent-pink/10 text-accent-pink">
          <FileText className="h-[18px] w-[18px]" strokeWidth={2.25} />
        </span>
        <div className="mt-4">
          <p className="text-2xl font-bold text-ink">
            {patternCount} <span className="text-ink-faint text-base font-medium">/ {sentenceCount} sent.</span>
          </p>
          <p className="text-sm text-ink-muted mt-0.5">Patterns flagged</p>
        </div>
      </div>
    </div>
  );
}
