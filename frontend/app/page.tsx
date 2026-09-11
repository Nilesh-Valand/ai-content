"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { getProject } from "@/lib/api";
import HumanizePanel from "@/components/HumanizePanel";
import { containsList, htmlToStructuredText } from "@/lib/pasteFormatting";
import { Wand2, Loader2, AlertCircle } from "lucide-react";

const PLACEHOLDER = `Paste a paragraph or two here — a blog post, an email draft, a report section — and get a natural, human-sounding rewrite.`;

function HumanizerPage() {
  const searchParams = useSearchParams();
  const projectParam = searchParams.get("project");

  const [content, setContent] = useState("");
  const [projectId, setProjectId] = useState<number | undefined>(undefined);
  const [initialHumanized, setInitialHumanized] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectParam) return;
    const id = Number(projectParam);
    if (!Number.isFinite(id)) return;

    setLoading(true);
    setError(null);
    getProject(id)
      .then((project) => {
        setContent(project.content);
        setProjectId(project.id);
        setInitialHumanized(project.humanized_content);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load that project."))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectParam]);

  const wordCount = content.trim().split(/\s+/).filter(Boolean).length;
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // A plain <textarea> only ever holds plain text, so pasting rich content
  // (a webpage, a Google Doc, ChatGPT output) goes through the browser's own
  // HTML-to-plain-text conversion first — and that's inconsistent about
  // whether bullet/number markers survive. Only step in when the clipboard
  // actually has a list in it, and hand-convert just that case to "- item" /
  // "1. item" text so it's never silently lost; anything else pastes normally.
  function handlePaste(e: React.ClipboardEvent<HTMLTextAreaElement>) {
    const html = e.clipboardData.getData("text/html");
    if (!html || !containsList(html)) return;

    e.preventDefault();
    const structured = htmlToStructuredText(html);
    const target = e.currentTarget;
    const start = target.selectionStart;
    const end = target.selectionEnd;
    const next = content.slice(0, start) + structured + content.slice(end);
    setContent(next);

    const cursor = start + structured.length;
    requestAnimationFrame(() => {
      textareaRef.current?.setSelectionRange(cursor, cursor);
    });
  }

  return (
    <main className="w-full bg-hero-gradient bg-no-repeat">
      <div className="w-full px-4 sm:px-8 lg:px-12 py-10 lg:py-14">
        <div className="mb-8 max-w-2xl">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 mb-4">
            <Wand2 className="h-3.5 w-3.5" strokeWidth={2.5} />
            Humanize Content
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-ink">
            Make it sound like you wrote it.
          </h1>
          <p className="text-ink-muted mt-3 text-[15px] leading-relaxed">
            Paste your content below and get a natural, human-sounding rewrite —
            varied sentence rhythm, concrete detail, less templated phrasing.
          </p>
        </div>

        <section className="rounded-2xl border border-border bg-surface shadow-card p-5 sm:p-6 mb-8">
          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onPaste={handlePaste}
            placeholder={PLACEHOLDER}
            rows={8}
            className="w-full bg-transparent outline-none text-[15px] leading-relaxed placeholder:text-ink-faint resize-y"
          />
          <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
            <span className="text-xs font-medium text-ink-faint">
              {wordCount} word{wordCount === 1 ? "" : "s"}
            </span>
          </div>
        </section>

        {loading && (
          <div className="flex items-center gap-2 text-sm text-ink-muted mb-8">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading…
          </div>
        )}

        {error && (
          <p className="flex items-center gap-2 text-sm text-danger-600 bg-danger-50 border border-danger-400/30 rounded-xl px-4 py-3 mb-8">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <HumanizePanel
          content={content}
          projectId={projectId}
          initialHumanized={initialHumanized}
        />
      </div>
    </main>
  );
}

export default function Home() {
  return (
    <Suspense fallback={null}>
      <HumanizerPage />
    </Suspense>
  );
}
