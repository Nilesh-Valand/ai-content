"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  TrainedPhrase,
  listTrainedPhrases,
  createTrainedPhrase,
  deleteTrainedPhrase,
} from "@/lib/api";

export default function TrainPage() {
  const [phrases, setPhrases] = useState<TrainedPhrase[]>([]);
  const [aiPhrase, setAiPhrase] = useState("");
  const [humanizedPhrase, setHumanizedPhrase] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    try {
      const res = await listTrainedPhrases();
      setPhrases(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load trained phrases.");
    } finally {
      setLoading(false);
    }
  }

  async function handleAdd() {
    if (!aiPhrase.trim() || !humanizedPhrase.trim()) return;
    setSaving(true);
    setError(null);
    try {
      const created = await createTrainedPhrase(aiPhrase, humanizedPhrase);
      setPhrases((prev) => [created, ...prev]);
      setAiPhrase("");
      setHumanizedPhrase("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete(id: number) {
    setDeletingId(id);
    setError(null);
    try {
      await deleteTrainedPhrase(id);
      setPhrases((prev) => prev.filter((p) => p.id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete phrase.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <main className="max-w-3xl mx-auto px-6 sm:px-8 py-14 sm:py-20">
      <header className="mb-10">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 font-mono text-xs uppercase tracking-[0.12em] border border-rule px-3 py-1.5 mb-6 text-ink-muted hover:text-ink hover:border-ink-faint transition-colors"
        >
          ← Back to Analyzer
        </Link>
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-pen-dim mb-3">
          Personal Style Training
        </p>
        <h1 className="font-display text-4xl sm:text-5xl leading-tight">
          Train Phrases
        </h1>
        <p className="font-body text-ink-muted mt-3 max-w-lg">
          Teach the humanizer your own voice. Pair an AI-sounding sentence with
          how you&rsquo;d actually rewrite it — an exact match gets swapped in
          verbatim, and every pair also guides the style of anything else it rewrites.
        </p>
      </header>

      <section className="mb-10 space-y-4">
        <div>
          <label className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted mb-2 block">
            AI-sounding phrase
          </label>
          <textarea
            value={aiPhrase}
            onChange={(e) => setAiPhrase(e.target.value)}
            placeholder="I like learning new things and trying to improve myself little by little."
            rows={3}
            className="w-full bg-transparent border border-rule focus:border-ink-faint outline-none px-4 py-3 font-body text-base leading-relaxed placeholder:text-ink-faint placeholder:italic resize-y"
          />
        </div>
        <div>
          <label className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted mb-2 block">
            Your humanized version
          </label>
          <textarea
            value={humanizedPhrase}
            onChange={(e) => setHumanizedPhrase(e.target.value)}
            placeholder="I enjoy picking up new things and getting a little better at them over time."
            rows={3}
            className="w-full bg-transparent border border-rule focus:border-ink-faint outline-none px-4 py-3 font-body text-base leading-relaxed placeholder:text-ink-faint placeholder:italic resize-y"
          />
        </div>
        <div className="flex justify-end">
          <button
            onClick={handleAdd}
            disabled={saving || !aiPhrase.trim() || !humanizedPhrase.trim()}
            className="font-mono text-xs uppercase tracking-[0.12em] bg-ink text-paper px-5 py-2.5 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-pen-dim transition-colors"
          >
            {saving ? "Saving…" : "Add Phrase"}
          </button>
        </div>
      </section>

      {error && (
        <p className="font-mono text-sm text-pen border border-pen bg-pen-soft px-4 py-3 mb-6">
          {error}
        </p>
      )}

      <section>
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted mb-4">
          Trained Pairs {phrases.length > 0 && `(${phrases.length})`}
        </p>

        {loading ? (
          <p className="font-body text-ink-muted italic">Loading…</p>
        ) : phrases.length === 0 ? (
          <p className="font-body text-ink-muted italic">
            No trained phrases yet. Add your first pair above.
          </p>
        ) : (
          <div className="space-y-6">
            {phrases.map((p) => (
              <div key={p.id} className="border-l-2 border-pen pl-5">
                <p className="font-body italic text-ink-muted text-sm mb-2">
                  &ldquo;{p.ai_phrase}&rdquo;
                </p>
                <p className="font-body text-base text-verdict mb-2">
                  <span className="font-mono text-[10px] uppercase tracking-wider mr-2">
                    Human
                  </span>
                  {p.humanized_phrase}
                </p>
                <button
                  onClick={() => handleDelete(p.id)}
                  disabled={deletingId === p.id}
                  className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-faint hover:text-pen transition-colors disabled:opacity-40"
                >
                  {deletingId === p.id ? "Removing…" : "Remove"}
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
