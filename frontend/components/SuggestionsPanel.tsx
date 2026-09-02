import { Suggestion } from "@/lib/api";

export default function SuggestionsPanel({ suggestions }: { suggestions: Suggestion[] }) {
  if (suggestions.length === 0) return null;

  return (
    <div className="fade-in">
      <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted mb-4">
        Suggested Rewrites
      </p>
      <div className="space-y-6">
        {suggestions.map((s, i) => (
          <div key={i} className="border-l-2 border-pen pl-5">
            {s.detected_sentence && (
              <p className="font-body italic text-ink-muted text-sm mb-2">
                &ldquo;{s.detected_sentence}&rdquo;
              </p>
            )}
            <p className="font-body text-base mb-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-pen-dim mr-2">
                Issue
              </span>
              {s.issue}
            </p>
            <p className="font-body text-base mb-1">
              <span className="font-mono text-[10px] uppercase tracking-wider text-ink-muted mr-2">
                Fix
              </span>
              {s.suggestion}
            </p>
            {s.improved_direction && (
              <p className="font-body text-base text-verdict">
                <span className="font-mono text-[10px] uppercase tracking-wider mr-2">
                  Try
                </span>
                {s.improved_direction}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
