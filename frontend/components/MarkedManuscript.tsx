import { SentenceScore } from "@/lib/api";

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
      <mark key={i} className="flag-mark bg-transparent text-pen-dim">
        {part}
      </mark>
    ) : (
      <span key={i}>{part}</span>
    );
  });
}

function likelihoodTint(pct: number) {
  if (pct >= 60) return "text-pen";
  if (pct >= 30) return "text-ink-muted";
  return "text-verdict";
}

export default function MarkedManuscript({
  sentences,
  highlightedPhrases,
}: {
  sentences: SentenceScore[];
  highlightedPhrases: string[];
}) {
  return (
    <div className="fade-in">
      <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted mb-4">
        Marked Manuscript
      </p>
      <div className="space-y-3">
        {sentences.map((s) => (
          <div key={s.index} className="flex gap-4 items-baseline group">
            <span className="font-mono text-[10px] text-ink-faint w-14 shrink-0 text-right pt-1">
              {s.ai_likelihood.toFixed(0)}%
            </span>
            <p className="font-body text-lg leading-relaxed">
              {markSentence(s.text, highlightedPhrases)}
            </p>
            <span
              className={`font-mono text-[10px] shrink-0 pt-1 ${likelihoodTint(
                s.ai_likelihood
              )} opacity-0 group-hover:opacity-100 transition-opacity hidden sm:inline`}
            >
              §{s.index}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
