"use client";

import { useEffect, useState } from "react";
import { humanizeContent, listProfiles, Profile } from "@/lib/api";
import { Wand2, Copy, Check, Loader2, Sparkles, AlertTriangle } from "lucide-react";

// navigator.clipboard requires a "secure context" (HTTPS, or the host machine's
// own localhost) — a browser on another machine reaching this app over plain
// HTTP via a LAN IP doesn't get it, so we fall back to the older execCommand
// approach, which has no such restriction.
function legacyCopy(text: string): boolean {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.style.position = "fixed";
  textarea.style.left = "-9999px";
  textarea.style.top = "0";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  document.body.removeChild(textarea);
  return ok;
}

async function copyToClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to legacy path
    }
  }
  return legacyCopy(text);
}

export default function HumanizePanel({
  content,
  projectId,
  initialHumanized,
}: {
  content: string;
  projectId?: number;
  initialHumanized?: string | null;
}) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState<number | undefined>(undefined);
  const [humanized, setHumanized] = useState<string | null>(initialHumanized ?? null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  useEffect(() => {
    listProfiles()
      .then((res) => {
        setProfiles(res);
        if (res.length > 0) {
          setSelectedProfileId(res[0].id);
        }
      })
      .catch(() => {
        // Degrade gracefully if profiles API unavailable
      });
  }, []);

  async function handleHumanize() {
    setLoading(true);
    setError(null);
    setCopied(false);
    try {
      const res = await humanizeContent(content, projectId, selectedProfileId);
      setHumanized(res.humanized_content);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  async function handleCopy() {
    if (!humanized) return;
    const ok = await copyToClipboard(humanized);
    if (ok) {
      setCopyFailed(false);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } else {
      setCopied(false);
      setCopyFailed(true);
      setTimeout(() => setCopyFailed(false), 2500);
    }
  }

  return (
    <div className="fade-in rounded-2xl border border-border bg-surface shadow-card p-6">
      <div className="flex items-center justify-between mb-4 gap-3 flex-wrap">
        <div className="flex items-center gap-3 flex-wrap">
          <p className="text-xs font-semibold uppercase tracking-wider text-ink-faint flex items-center gap-1.5">
            <Wand2 className="h-3.5 w-3.5" strokeWidth={2.5} />
            Humanize Content
          </p>

          {profiles.length > 0 && (
            <div className="flex items-center gap-1.5 bg-surface-muted border border-border rounded-lg px-2.5 py-1 text-xs">
              <Sparkles className="h-3.5 w-3.5 text-brand-600 shrink-0" />
              <span className="font-semibold text-ink-muted">Style Profile:</span>
              <select
                value={selectedProfileId ?? ""}
                onChange={(e) => setSelectedProfileId(Number(e.target.value))}
                className="bg-transparent text-ink font-medium outline-none cursor-pointer"
              >
                {profiles.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name} ({p.phrase_count} phrases)
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <button
          onClick={handleHumanize}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2 shadow-lift disabled:opacity-50 disabled:cursor-not-allowed hover:brightness-110 transition"
        >
          {loading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Wand2 className="h-4 w-4" />
          )}
          {loading ? "Rewriting…" : humanized ? "Regenerate" : "Humanize Content"}
        </button>
      </div>

      {error && (
        <p className="text-sm text-danger-600 bg-danger-50 border border-danger-400/30 rounded-lg px-4 py-3 mb-4">
          {error}
        </p>
      )}

      {loading && !humanized && (
        <div className="rounded-xl border border-border bg-surface-muted p-4 space-y-2 animate-pulse-glow">
          <div className="h-3.5 bg-border-strong rounded w-11/12" />
          <div className="h-3.5 bg-border-strong rounded w-10/12" />
          <div className="h-3.5 bg-border-strong rounded w-8/12" />
        </div>
      )}

      {humanized && (
        <div className="rounded-xl border border-success-400/30 bg-success-50/40 p-4">
          <p className="text-[15px] leading-relaxed whitespace-pre-wrap text-ink">
            {humanized}
          </p>
          <div className="flex items-center justify-end gap-2 mt-3">
            {copyFailed && (
              <span className="flex items-center gap-1 text-xs text-danger-600">
                <AlertTriangle className="h-3.5 w-3.5" />
                Couldn&rsquo;t copy — select the text and copy manually
              </span>
            )}
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-surface px-2.5 py-1.5 text-xs font-medium text-ink-muted hover:text-ink hover:border-border-strong transition-colors"
            >
              {copied ? (
                <Check className="h-3.5 w-3.5 text-success-600" />
              ) : (
                <Copy className="h-3.5 w-3.5" />
              )}
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
