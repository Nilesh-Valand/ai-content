import { Suggestion } from "@/lib/api";
import { Lightbulb, Quote, Wrench, ArrowRight } from "lucide-react";

export default function SuggestionsPanel({ suggestions }: { suggestions: Suggestion[] }) {
  if (suggestions.length === 0) return null;

  return (
    <div className="fade-in rounded-2xl border border-border bg-surface shadow-card p-6">
      <p className="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-4 flex items-center gap-1.5">
        <Lightbulb className="h-3.5 w-3.5" strokeWidth={2.5} />
        Suggested Rewrites
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {suggestions.map((s, i) => (
          <div
            key={i}
            className="rounded-xl border border-border bg-surface-muted p-4 flex flex-col gap-2.5"
          >
            {s.detected_sentence && (
              <p className="flex items-start gap-1.5 text-sm text-ink-muted italic">
                <Quote className="h-3.5 w-3.5 shrink-0 mt-0.5 text-ink-faint" />
                {s.detected_sentence}
              </p>
            )}
            <div className="flex items-start gap-1.5 text-sm text-ink">
              <span className="shrink-0 mt-0.5 inline-flex h-4 w-4 items-center justify-center rounded-full bg-warn-50 text-warn-600 text-[10px] font-bold">
                !
              </span>
              {s.issue}
            </div>
            <div className="flex items-start gap-1.5 text-sm text-ink">
              <Wrench className="h-3.5 w-3.5 shrink-0 mt-0.5 text-brand-500" />
              {s.suggestion}
            </div>
            {s.improved_direction && (
              <div className="flex items-start gap-1.5 text-sm font-medium text-success-600 border-t border-border pt-2.5 mt-0.5">
                <ArrowRight className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                {s.improved_direction}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
