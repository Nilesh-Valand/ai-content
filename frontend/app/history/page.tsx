"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ProjectSummary, listProjects, deleteProject } from "@/lib/api";
import {
  History,
  Loader2,
  AlertCircle,
  FolderOpen,
  Trash2,
  Wand2,
  Check,
  X,
  ArrowRight,
} from "lucide-react";

function formatDate(iso: string) {
  const d = new Date(iso.replace(" ", "T") + "Z");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function verdictTheme(pct: number) {
  if (pct >= 60) return { chip: "bg-danger-50 text-danger-600", label: "Likely AI" };
  if (pct >= 30) return { chip: "bg-warn-50 text-warn-600", label: "Mixed" };
  return { chip: "bg-success-50 text-success-600", label: "Likely Human" };
}

export default function HistoryPage() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [confirmingId, setConfirmingId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [bulkConfirming, setBulkConfirming] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const selectAllRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (selectAllRef.current) {
      selectAllRef.current.indeterminate = selectedIds.size > 0 && selectedIds.size < projects.length;
    }
  }, [selectedIds, projects.length]);

  function toggleSelect(id: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function toggleSelectAll() {
    setSelectedIds((prev) =>
      prev.size === projects.length ? new Set() : new Set(projects.map((p) => p.id))
    );
  }

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const res = await listProjects();
      setProjects(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load history.");
    } finally {
      setLoading(false);
    }
  }

  async function handleDelete(id: number) {
    setDeletingId(id);
    setError(null);
    try {
      await deleteProject(id);
      setProjects((prev) => prev.filter((p) => p.id !== id));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete project.");
    } finally {
      setDeletingId(null);
      setConfirmingId(null);
    }
  }

  async function handleBulkDelete() {
    setBulkDeleting(true);
    setError(null);
    const ids = Array.from(selectedIds);
    const results = await Promise.allSettled(ids.map((id) => deleteProject(id)));
    const failedIds = ids.filter((_, i) => results[i].status === "rejected");
    const succeeded = new Set(ids.filter((id) => !failedIds.includes(id)));

    setProjects((prev) => prev.filter((p) => !succeeded.has(p.id)));
    setSelectedIds(new Set(failedIds));
    if (failedIds.length > 0) {
      setError(
        `Could not delete ${failedIds.length} item${failedIds.length === 1 ? "" : "s"}. Try again.`
      );
    }
    setBulkDeleting(false);
    setBulkConfirming(false);
  }

  return (
    <main className="w-full bg-hero-gradient bg-no-repeat">
      <div className="w-full px-4 sm:px-8 lg:px-12 py-10 lg:py-14">
        <div className="max-w-2xl mb-8">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 mb-4">
            <History className="h-3.5 w-3.5" strokeWidth={2.5} />
            Saved Analyses
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-ink">
            History
          </h1>
          <p className="text-ink-muted mt-3 text-[15px] leading-relaxed">
            Every content analysis is saved automatically. Reopen one to see its
            full breakdown and humanized version again.
          </p>
        </div>

        {error && (
          <p className="flex items-center gap-2 text-sm text-danger-600 bg-danger-50 border border-danger-400/30 rounded-xl px-4 py-3 mb-6">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        {!loading && projects.length > 0 && (
          <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
            <label className="flex items-center gap-2 text-sm text-ink-muted cursor-pointer select-none">
              <input
                ref={selectAllRef}
                type="checkbox"
                checked={selectedIds.size > 0 && selectedIds.size === projects.length}
                onChange={toggleSelectAll}
                className="h-4 w-4 accent-brand-600 cursor-pointer"
              />
              {selectedIds.size > 0 ? `${selectedIds.size} selected` : "Select all"}
            </label>

            {selectedIds.size > 0 &&
              (bulkConfirming ? (
                <div className="flex items-center gap-2">
                  <span className="text-sm text-ink-muted">
                    Delete {selectedIds.size} item{selectedIds.size === 1 ? "" : "s"}?
                  </span>
                  <button
                    onClick={handleBulkDelete}
                    disabled={bulkDeleting}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-danger-500 hover:bg-danger-600 text-white px-3 py-1.5 text-sm font-semibold transition-colors disabled:opacity-50"
                  >
                    {bulkDeleting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Check className="h-4 w-4" />
                    )}
                    Confirm
                  </button>
                  <button
                    onClick={() => setBulkConfirming(false)}
                    className="rounded-lg p-1.5 text-ink-muted hover:text-ink hover:bg-surface-muted transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setBulkConfirming(true)}
                  className="inline-flex items-center gap-2 rounded-lg border border-danger-400/30 bg-danger-50 text-danger-600 hover:bg-danger-100 px-3.5 py-2 text-sm font-semibold transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete selected
                </button>
              ))}
          </div>
        )}

        {loading ? (
          <div className="rounded-2xl border border-border bg-surface shadow-card p-10 flex items-center justify-center gap-2 text-ink-muted">
            <Loader2 className="h-5 w-5 animate-spin" />
            Loading history…
          </div>
        ) : projects.length === 0 ? (
          <div className="rounded-2xl border border-border bg-surface shadow-card p-14 flex flex-col items-center text-center gap-3">
            <span className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-600">
              <FolderOpen className="h-6 w-6" strokeWidth={2} />
            </span>
            <p className="text-ink font-semibold">No saved analyses yet</p>
            <p className="text-sm text-ink-muted max-w-sm">
              Analyze some content on the home page and it will show up here
              automatically.
            </p>
            <Link
              href="/"
              className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2.5 shadow-lift hover:brightness-110 transition mt-2"
            >
              Go to Analyzer
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {projects.map((p) => {
              const theme = verdictTheme(p.ai_writing_likelihood);
              const selected = selectedIds.has(p.id);
              return (
                <div
                  key={p.id}
                  className={`rounded-2xl border shadow-card p-5 flex flex-col gap-3 transition-colors ${
                    selected ? "border-brand-300 ring-2 ring-brand-100 bg-surface" : "border-border bg-surface"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2.5">
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleSelect(p.id)}
                        className="h-4 w-4 accent-brand-600 cursor-pointer shrink-0"
                      />
                      <span
                        className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${theme.chip}`}
                      >
                        {p.ai_writing_likelihood.toFixed(0)}% · {theme.label}
                      </span>
                    </div>
                    {p.has_humanized && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-brand-50 text-brand-600 px-2.5 py-1 text-xs font-semibold shrink-0">
                        <Wand2 className="h-3 w-3" strokeWidth={2.5} />
                        Humanized
                      </span>
                    )}
                  </div>

                  <p className="text-sm text-ink line-clamp-4 leading-relaxed flex-1">
                    {p.content}
                  </p>

                  <div className="flex items-center justify-between pt-3 border-t border-border">
                    <span className="text-xs text-ink-faint">{formatDate(p.created_at)}</span>
                    <div className="flex items-center gap-1.5">
                      {confirmingId === p.id ? (
                        <>
                          <span className="text-xs text-ink-muted mr-1">Delete?</span>
                          <button
                            onClick={() => handleDelete(p.id)}
                            disabled={deletingId === p.id}
                            className="rounded-md p-1.5 text-white bg-danger-500 hover:bg-danger-600 transition-colors disabled:opacity-50"
                            title="Confirm delete"
                          >
                            {deletingId === p.id ? (
                              <Loader2 className="h-3.5 w-3.5 animate-spin" />
                            ) : (
                              <Check className="h-3.5 w-3.5" />
                            )}
                          </button>
                          <button
                            onClick={() => setConfirmingId(null)}
                            className="rounded-md p-1.5 text-ink-muted hover:text-ink hover:bg-surface-muted transition-colors"
                            title="Cancel"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </>
                      ) : (
                        <>
                          <button
                            onClick={() => setConfirmingId(p.id)}
                            className="rounded-md p-1.5 text-ink-muted hover:text-danger-600 hover:bg-danger-50 transition-colors"
                            title="Delete"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                          <Link
                            href={`/?project=${p.id}`}
                            className="inline-flex items-center gap-1 rounded-md bg-brand-50 text-brand-700 hover:bg-brand-100 px-2.5 py-1.5 text-xs font-semibold transition-colors"
                          >
                            Open
                            <ArrowRight className="h-3 w-3" />
                          </Link>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
