"use client";

import { useState } from "react";
import { analyzeContent, AnalyzeResponse } from "@/lib/api";
import VerdictStamp from "@/components/VerdictStamp";
import MarkedManuscript from "@/components/MarkedManuscript";
import PatternNotes from "@/components/PatternNotes";
import SuggestionsPanel from "@/components/SuggestionsPanel";

const PLACEHOLDER = `Paste a paragraph or two here — a blog post, an email draft, a report section — and I'll mark it up the way an editor would.`;

export default function Home() {
  const [content, setContent] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyze() {
    if (!content.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeContent(content);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-3xl mx-auto px-6 sm:px-8 py-14 sm:py-20">
      <header className="mb-10">
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-pen-dim mb-3">
          Manuscript Review
        </p>
        <h1 className="font-display text-4xl sm:text-5xl leading-tight">
          AI Content Detector
        </h1>
        <p className="font-body text-ink-muted mt-3 max-w-lg">
          A rule-based readout of vocabulary, structure, and specificity —
          the same signals a careful editor watches for.
        </p>
      </header>

      <section className="mb-6">
        <textarea
          value={content}
          onChange={(e) => setContent(e.target.value)}
          placeholder={PLACEHOLDER}
          rows={10}
          className="w-full bg-transparent border border-rule focus:border-ink-faint outline-none px-4 py-4 font-body text-lg leading-relaxed placeholder:text-ink-faint placeholder:italic resize-y"
        />
        <div className="flex items-center justify-between mt-3">
          <span className="font-mono text-xs text-ink-faint">
            {content.trim().split(/\s+/).filter(Boolean).length} words
          </span>
          <button
            onClick={handleAnalyze}
            disabled={loading || !content.trim()}
            className="font-mono text-xs uppercase tracking-[0.12em] bg-ink text-paper px-5 py-2.5 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-pen-dim transition-colors"
          >
            {loading ? "Reviewing…" : "Analyze Content"}
          </button>
        </div>
      </section>

      {error && (
        <p className="font-mono text-sm text-pen border border-pen bg-pen-soft px-4 py-3 mb-6">
          {error}
        </p>
      )}

      {result && (
        <div className="space-y-10">
          <VerdictStamp
            percentage={result.overall.ai_writing_likelihood}
            confidence={result.overall.confidence}
          />
          <PatternNotes patterns={result.detected_patterns} />
          <MarkedManuscript
            sentences={result.sentence_scores}
            highlightedPhrases={result.highlighted_phrases}
          />
          <SuggestionsPanel suggestions={result.suggestions} />
        </div>
      )}
    </main>
  );
}
