"use client";

import { useState } from "react";
import { humanizeContent } from "@/lib/api";

export default function HumanizePanel({ content }: { content: string }) {
  const [humanized, setHumanized] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  async function handleHumanize() {
    setLoading(true);
    setError(null);
    setCopied(false);
    try {
      const res = await humanizeContent(content);
      setHumanized(res.humanized_content);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  async function handleCopy() {
    if (!humanized) return;
    try {
      await navigator.clipboard.writeText(humanized);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable — ignore
    }
  }

  return (
    <div className="fade-in">
      <div className="flex items-center justify-between mb-4">
        <p className="font-mono text-[11px] tracking-[0.18em] uppercase text-ink-muted">
          Humanize Content
        </p>
        <button
          onClick={handleHumanize}
          disabled={loading}
          className="font-mono text-xs uppercase tracking-[0.12em] bg-ink text-paper px-5 py-2.5 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-pen-dim transition-colors"
        >
          {loading ? "Rewriting…" : humanized ? "Regenerate" : "Humanize Content"}
        </button>
      </div>

      {error && (
        <p className="font-mono text-sm text-pen border border-pen bg-pen-soft px-4 py-3 mb-4">
          {error}
        </p>
      )}

      {humanized && (
        <div className="border border-rule px-4 py-4">
          <p className="font-body text-lg leading-relaxed whitespace-pre-wrap">
            {humanized}
          </p>
          <div className="flex justify-end mt-3">
            <button
              onClick={handleCopy}
              className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-muted hover:text-ink transition-colors"
            >
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
