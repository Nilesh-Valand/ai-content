import { SentenceScore } from "@/lib/api";
import { ScrollText } from "lucide-react";

function escapeRegExp(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function markSentence(text: string, phrases: string[]) {
  if (phrases.length === 0) return text;
  const sorted = [...phrases].sort((a, b) => b.length - a.length).map(escapeRegExp);
  const pattern = new RegExp(`(${sorted.join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, i) => {
    const isMatch = phrases.some((p) => p.toLowerCase() === part.toLowerCase());
    return isMatch ? (
      <mark key={i} className="flag-mark bg-transparent text-danger-600 font-medium">
        {part}
      </mark>
    ) : (
      <span key={i}>{part}</span>
    );
  });
}

function likelihoodChip(pct: number) {
  if (pct >= 60) return "bg-danger-50 text-danger-600";
  if (pct >= 30) return "bg-warn-50 text-warn-600";
  return "bg-success-50 text-success-600";
}

export default function MarkedManuscript({
  sentences,
  highlightedPhrases,
}: {
  sentences: SentenceScore[];
  highlightedPhrases: string[];
}) {
  return (
    <div className="fade-in rounded-2xl border border-border bg-surface shadow-card p-6 h-full">
      <p className="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-4 flex items-center gap-1.5">
        <ScrollText className="h-3.5 w-3.5" strokeWidth={2.5} />
        Sentence-by-Sentence Markup
      </p>
      <div className="space-y-1 max-h-[520px] overflow-y-auto pr-1">
        {sentences.map((s) => (
          <div
            key={s.index}
            className="flex gap-3 items-baseline rounded-lg px-2 py-2 hover:bg-surface-muted transition-colors"
          >
            <span
              className={`font-mono text-[10px] font-semibold shrink-0 rounded-full px-1.5 py-0.5 w-11 text-center ${likelihoodChip(
                s.ai_likelihood
              )}`}
            >
              {s.ai_likelihood.toFixed(0)}%
            </span>
            <p className="text-[15px] leading-relaxed text-ink">
              {markSentence(s.text, highlightedPhrases)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
