import { Confidence } from "@/lib/api";

function verdictColor(pct: number) {
  if (pct >= 60) return { text: "text-pen", ring: "border-pen", soft: "bg-pen-soft" };
  if (pct >= 30) return { text: "text-ink", ring: "border-ink-faint", soft: "bg-paper-dim" };
  return { text: "text-verdict", ring: "border-verdict", soft: "bg-verdict-soft" };
}

export default function VerdictStamp({
  percentage,
  confidence,
}: {
  percentage: number;
  confidence: Confidence;
}) {
  const c = verdictColor(percentage);
  return (
    <div className="fade-in flex flex-col sm:flex-row sm:items-end gap-4 sm:gap-8 py-6 border-y rule-hairline">
      <div>
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted mb-1">
          AI-Writing Likelihood
        </p>
        <p className={`font-mono text-6xl sm:text-7xl leading-none ${c.text}`}>
          {percentage.toFixed(0)}
          <span className="text-2xl align-top ml-1">%</span>
        </p>
      </div>
      <div className={`inline-flex items-center gap-2 border ${c.ring} ${c.soft} px-3 py-1.5 self-start sm:self-end mb-1`}>
        <span className="font-mono text-[11px] tracking-[0.12em] uppercase text-ink-muted">
          Confidence
        </span>
        <span className="font-mono text-[11px] tracking-[0.12em] uppercase font-semibold">
          {confidence}
        </span>
      </div>
      <p className="font-body italic text-ink-muted text-sm sm:ml-auto max-w-xs">
        A likelihood estimate from writing-pattern analysis — not proof the text was AI-written.
      </p>
    </div>
  );
}
