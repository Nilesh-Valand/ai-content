"use client";

import { useEffect, useRef, useState } from "react";
import { humanizeContent, listProfiles, listTrainedPhrases, Profile, TrainedPhrase } from "@/lib/api";
import { Wand2, Copy, Check, Loader2, Sparkles, AlertTriangle, ChevronDown, Loader } from "lucide-react";
import { useDismiss } from "@/hooks/useDismiss";
import { useUser } from "@/contexts/UserContext";

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
  const [selectedPhraseIds, setSelectedPhraseIds] = useState<number[]>([]);
  const [humanized, setHumanized] = useState<string | null>(initialHumanized ?? null);
  // Starts from the projectId prop (set when a project was loaded from
  // History via its URL), but updates from the server's response after a
  // fresh humanize with no prop — the backend creates a project on the
  // fly in that case (see main.py's /humanize handler) so it still shows
  // up in History, and this makes sure a later "Regenerate" in the same
  // session updates that same entry instead of creating a new one each time.
  const [currentProjectId, setCurrentProjectId] = useState<number | undefined>(projectId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);

  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false);
  const [phrasesByProfile, setPhrasesByProfile] = useState<Record<number, TrainedPhrase[]>>({});
  const [loadingProfileId, setLoadingProfileId] = useState<number | null>(null);

  const profileDropdownRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  function closeProfileDropdown() {
    setProfileDropdownOpen(false);
  }
  useDismiss(profileDropdownRef, profileDropdownOpen, closeProfileDropdown);

  const selectedProfile = profiles.find((p) => p.id === selectedProfileId);
  // The right-hand panel previews whichever profile is actually selected —
  // click a profile row to preview it (tap/click is far more reliable than
  // hover here: hover-driven previews fought with the auto-scroll below,
  // since scrolling the page mid-hover moves the row out from under the
  // cursor and breaks hover tracking; a click is a single discrete event
  // that doesn't have that problem, and it works on touch devices too).
  const previewProfileId = selectedProfileId ?? null;
  const previewProfile = profiles.find((p) => p.id === previewProfileId);
  const previewPhrases = previewProfileId !== null ? phrasesByProfile[previewProfileId] : undefined;

  // The popover's height changes as you switch profiles and as each one's
  // phrases finish loading — re-check that it's fully in view every time
  // that happens, not just once when it first opens, otherwise a box that
  // grows after its data loads can end up cut off with no further correction.
  useEffect(() => {
    if (profileDropdownOpen) {
      popoverRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [profileDropdownOpen, previewProfileId, previewPhrases]);

  const { activeUserId } = useUser();

  // The parent only sets the projectId prop when a project was loaded
  // from History (via its URL) — pick that up if it changes, e.g.
  // navigating from one History entry to another without a full reload.
  useEffect(() => {
    setCurrentProjectId(projectId);
  }, [projectId]);

  useEffect(() => {
    if (activeUserId === undefined) return;
    listProfiles(activeUserId)
      .then((res) => {
        setProfiles(res);
        setPhrasesByProfile({});
        setSelectedProfileId(res[0]?.id);
        setSelectedPhraseIds([]);
      })
      .catch(() => {
        // Degrade gracefully if profiles API unavailable
      });
  }, [activeUserId]);

  useEffect(() => {
    if (previewProfileId === null || phrasesByProfile[previewProfileId] !== undefined) return;
    setLoadingProfileId(previewProfileId);
    listTrainedPhrases(previewProfileId)
      .then((res) => setPhrasesByProfile((prev) => ({ ...prev, [previewProfileId]: res })))
      .catch(() => setPhrasesByProfile((prev) => ({ ...prev, [previewProfileId]: [] })))
      .finally(() => setLoadingProfileId(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewProfileId]);

  function selectProfile(profileId: number) {
    if (profileId !== selectedProfileId) {
      setSelectedProfileId(profileId);
      setSelectedPhraseIds([]);
    }
  }

  function togglePhrase(phraseId: number) {
    setSelectedPhraseIds((prev) =>
      prev.includes(phraseId) ? prev.filter((p) => p !== phraseId) : [...prev, phraseId]
    );
  }

  function selectAllPhrases() {
    if (selectedProfileId === undefined) return;
    setSelectedPhraseIds((phrasesByProfile[selectedProfileId] ?? []).map((p) => p.id));
  }

  function clearPhrases() {
    setSelectedPhraseIds([]);
  }

  async function handleHumanize() {
    setLoading(true);
    setError(null);
    setCopied(false);
    try {
      const res = await humanizeContent(
        content,
        currentProjectId,
        selectedProfileId,
        selectedPhraseIds,
        activeUserId
      );
      setHumanized(res.humanized_content);
      if (typeof res.project_id === "number") {
        setCurrentProjectId(res.project_id);
      }
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
            <div className="relative" ref={profileDropdownRef}>
              <button
                onClick={() => {
                  if (profileDropdownOpen) {
                    closeProfileDropdown();
                  } else {
                    setProfileDropdownOpen(true);
                  }
                }}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface-muted px-2.5 py-1.5 text-xs font-medium text-ink hover:border-border-strong transition-colors"
              >
                <Sparkles className="h-3.5 w-3.5 text-brand-600 shrink-0" />
                <span className="text-ink-muted font-semibold">Style Profile:</span>
                <span className="font-semibold">{selectedProfile?.name ?? "Select…"}</span>
                {selectedPhraseIds.length > 0 && (
                  <span className="rounded-full bg-brand-100 text-brand-700 px-1.5 py-0.5 text-[10px] font-semibold">
                    {selectedPhraseIds.length} picked
                  </span>
                )}
                <ChevronDown className={`h-3.5 w-3.5 text-ink-faint transition-transform ${profileDropdownOpen ? "rotate-180" : ""}`} />
              </button>

              {profileDropdownOpen && (
                <div
                  ref={popoverRef}
                  className="fade-in absolute left-0 z-20 mt-2 flex max-w-[calc(100vw-2rem)] overflow-hidden rounded-2xl border border-border bg-surface shadow-card flex-col sm:flex-row"
                >
                  {/* Left: profile list */}
                  <ul className="w-full sm:w-64 shrink-0 border-b sm:border-b-0 sm:border-r border-border max-h-56 sm:max-h-[28rem] overflow-y-auto py-2">
                    {profiles.map((p) => {
                      const active = p.id === selectedProfileId;
                      return (
                        <li key={p.id}>
                          <button
                            onClick={() => selectProfile(p.id)}
                            className={`flex w-full items-center justify-between gap-2 px-4 py-3 text-left text-[15px] transition-colors ${
                              active ? "bg-brand-50 text-brand-700" : "text-ink hover:bg-surface-muted"
                            }`}
                          >
                            <span className="flex items-center gap-2.5 min-w-0">
                              <Sparkles className={`h-4 w-4 shrink-0 ${active ? "text-brand-600" : "text-ink-faint"}`} />
                              <span className="truncate font-medium">{p.name}</span>
                              <span className="shrink-0 rounded-full bg-surface-muted border border-border px-2 py-0.5 text-xs font-semibold text-ink-faint">
                                {p.phrase_count}
                              </span>
                            </span>
                            {active && <Check className="h-4 w-4 shrink-0 text-brand-600" />}
                          </button>
                        </li>
                      );
                    })}
                  </ul>

                  {/* Right: phrase flyout for the selected profile */}
                  <div className="w-full sm:w-[26rem] flex flex-col max-h-[28rem]">
                    <div className="flex items-center justify-between gap-2 px-5 py-3.5 border-b border-border bg-surface-muted shrink-0">
                      <p className="text-sm font-semibold text-ink truncate">
                        {previewProfile?.name ?? "Phrases"}
                      </p>
                      {(previewPhrases?.length ?? 0) > 0 && (
                        <div className="flex items-center gap-3 shrink-0">
                          <button
                            onClick={selectAllPhrases}
                            className="text-xs font-semibold text-brand-600 hover:text-brand-700"
                          >
                            Select all
                          </button>
                          <button
                            onClick={clearPhrases}
                            className="text-xs font-semibold text-ink-muted hover:text-ink"
                          >
                            Clear
                          </button>
                        </div>
                      )}
                    </div>

                    {loadingProfileId === previewProfileId ? (
                      <div className="flex items-center gap-2 text-sm text-ink-muted px-5 py-8">
                        <Loader className="h-4 w-4 animate-spin" /> Loading phrases…
                      </div>
                    ) : !previewPhrases || previewPhrases.length === 0 ? (
                      <p className="px-5 py-8 text-center text-sm text-ink-faint">
                        No trained phrases in this profile yet.
                      </p>
                    ) : (
                      <ul className="overflow-y-auto flex-1 divide-y divide-border">
                        {previewPhrases.map((ph) => {
                          const checked = selectedPhraseIds.includes(ph.id);
                          return (
                            <li key={ph.id}>
                              <label className="flex items-start gap-3 px-5 py-3.5 cursor-pointer hover:bg-surface-muted transition-colors">
                                <input
                                  type="checkbox"
                                  checked={checked}
                                  onChange={() => togglePhrase(ph.id)}
                                  className="mt-1 h-4 w-4 shrink-0 accent-brand-600 cursor-pointer"
                                />
                                <span className="min-w-0 text-sm leading-relaxed">
                                  <span className="block text-ink-muted italic line-clamp-2">
                                    &ldquo;{ph.ai_phrase}&rdquo;
                                  </span>
                                  <span className="block text-ink font-medium line-clamp-2 mt-1">
                                    {ph.humanized_phrase}
                                  </span>
                                </span>
                              </label>
                            </li>
                          );
                        })}
                      </ul>
                    )}

                    <p className="px-5 py-3 text-xs text-ink-faint border-t border-border shrink-0">
                      {selectedPhraseIds.length === 0
                        ? "None picked — AI auto-picks the most relevant phrases."
                        : `Using ${selectedPhraseIds.length} picked phrase${selectedPhraseIds.length === 1 ? "" : "s"}.`}
                    </p>
                  </div>
                </div>
              )}
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
