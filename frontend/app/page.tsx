"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { analyzeContent, getProject, AnalyzeResponse } from "@/lib/api";
import { useUser } from "@/contexts/UserContext";
import VerdictStamp from "@/components/VerdictStamp";
import MarkedManuscript from "@/components/MarkedManuscript";
import PatternNotes from "@/components/PatternNotes";
import SuggestionsPanel from "@/components/SuggestionsPanel";
import HumanizePanel from "@/components/HumanizePanel";
import { ScanSearch, Loader2, AlertCircle } from "lucide-react";

const PLACEHOLDER = `Paste a paragraph or two here — a blog post, an email draft, a report section — and get an instant AI-writing likelihood breakdown.`;

function AnalyzerPage() {
  const searchParams = useSearchParams();
  const projectParam = searchParams.get("project");

  const [content, setContent] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [initialHumanized, setInitialHumanized] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { activeUserId } = useUser();

  useEffect(() => {
    if (!projectParam) return;
    const id = Number(projectParam);
    if (!Number.isFinite(id)) return;

    setLoading(true);
    setError(null);
    getProject(id)
      .then((project) => {
        setContent(project.content);
        setInitialHumanized(project.humanized_content);
        setResult({
          id: project.id,
          overall: project.overall,
          detected_patterns: project.detected_patterns,
          sentence_scores: project.sentence_scores,
          highlighted_phrases: project.highlighted_phrases,
          suggestions: project.suggestions,
        });
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load that project."))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectParam]);

  async function handleAnalyze() {
    if (!content.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeContent(content, true, activeUserId);
      setInitialHumanized(null);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  const wordCount = content.trim().split(/\s+/).filter(Boolean).length;

  return (
    <main className="w-full bg-hero-gradient bg-no-repeat">
      <div className="w-full px-4 sm:px-8 lg:px-12 py-10 lg:py-14">
        <div className="mb-8 max-w-2xl">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 mb-4">
            <ScanSearch className="h-3.5 w-3.5" strokeWidth={2.5} />
            AI Content Detector
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-ink">
            Know what&rsquo;s AI. Fix what isn&rsquo;t you.
          </h1>
          <p className="text-ink-muted mt-3 text-[15px] leading-relaxed">
            A rule-based readout of vocabulary, structure, and specificity — the
            same signals a careful editor watches for — with AI-powered rewrite
            and humanize tools built in. Every analysis is saved automatically —
            find it later in History.
          </p>
        </div>

        <section className="rounded-2xl border border-border bg-surface shadow-card p-5 sm:p-6 mb-8">
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder={PLACEHOLDER}
            rows={8}
            className="w-full bg-transparent outline-none text-[15px] leading-relaxed placeholder:text-ink-faint resize-y"
          />
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
            <span className="text-xs font-medium text-ink-faint">
              {wordCount} word{wordCount === 1 ? "" : "s"}
            </span>
            <button
              onClick={handleAnalyze}
              disabled={loading || !content.trim()}
              className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-5 py-2.5 shadow-lift disabled:opacity-50 disabled:cursor-not-allowed hover:brightness-110 transition"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <ScanSearch className="h-4 w-4" />
              )}
              {loading ? "Analyzing…" : "Analyze Content"}
            </button>
          </div>
        </section>

        {error && (
          <p className="flex items-center gap-2 text-sm text-danger-600 bg-danger-50 border border-danger-400/30 rounded-xl px-4 py-3 mb-8">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        {result && (
          <div className="space-y-6" key={result.id}>
            <VerdictStamp
              percentage={result.overall.ai_writing_likelihood}
              confidence={result.overall.confidence}
              patternCount={
                result.detected_patterns.filter((p) => p.severity !== "Low").length
              }
              sentenceCount={result.sentence_scores.length}
            />

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-5">
                <PatternNotes patterns={result.detected_patterns} />
              </div>
              <div className="lg:col-span-7">
                <MarkedManuscript
                  sentences={result.sentence_scores}
                  highlightedPhrases={result.highlighted_phrases}
                />
              </div>
            </div>

            <SuggestionsPanel suggestions={result.suggestions} />
            <HumanizePanel
              content={content}
              projectId={result.id}
              initialHumanized={initialHumanized}
            />
          </div>
        )}
      </div>
    </main>
  );
}

export default function Home() {
  return (
    <Suspense fallback={null}>
      <AnalyzerPage />
    </Suspense>
  );
}
