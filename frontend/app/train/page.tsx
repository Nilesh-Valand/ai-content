"use client";

import { useEffect, useState } from "react";
import {
  TrainedPhrase,
  listTrainedPhrases,
  createTrainedPhrase,
  updateTrainedPhrase,
  deleteTrainedPhrase,
} from "@/lib/api";
import Modal from "@/components/Modal";
import {
  GraduationCap,
  Plus,
  Pencil,
  Trash2,
  Loader2,
  AlertCircle,
  BookOpen,
  Check,
  X,
} from "lucide-react";

type ModalState = { mode: "create" } | { mode: "edit"; phrase: TrainedPhrase } | null;

function formatDate(iso: string) {
  const d = new Date(iso.replace(" ", "T") + "Z");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export default function TrainPage() {
  const [phrases, setPhrases] = useState<TrainedPhrase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modal, setModal] = useState<ModalState>(null);
  const [confirmingId, setConfirmingId] = useState<number | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);

  useEffect(() => {
    refresh();
  }, []);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const res = await listTrainedPhrases();
      setPhrases(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load trained phrases.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSave(aiPhrase: string, humanizedPhrase: string) {
    if (modal?.mode === "edit") {
      const updated = await updateTrainedPhrase(modal.phrase.id, aiPhrase, humanizedPhrase);
      setPhrases((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
    } else {
      const created = await createTrainedPhrase(aiPhrase, humanizedPhrase);
      setPhrases((prev) => [created, ...prev]);
    }
    setModal(null);
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
      setConfirmingId(null);
    }
  }

  return (
    <main className="w-full bg-hero-gradient bg-no-repeat">
      <div className="w-full px-4 sm:px-8 lg:px-12 py-10 lg:py-14">
        <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 mb-4">
              <GraduationCap className="h-3.5 w-3.5" strokeWidth={2.5} />
              Personal Style Training
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-ink">
              Train Phrases
            </h1>
            <p className="text-ink-muted mt-3 text-[15px] leading-relaxed">
              Teach the humanizer your own voice. An exact match gets swapped in
              verbatim, and every pair also guides the style of anything else it rewrites.
            </p>
          </div>
          <button
            onClick={() => setModal({ mode: "create" })}
            className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2.5 shadow-lift hover:brightness-110 transition shrink-0"
          >
            <Plus className="h-4 w-4" strokeWidth={2.5} />
            Add Phrase
          </button>
        </div>

        {error && (
          <p className="flex items-center gap-2 text-sm text-danger-600 bg-danger-50 border border-danger-400/30 rounded-xl px-4 py-3 mb-6">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <div className="rounded-2xl border border-border bg-surface shadow-card overflow-hidden">
          {loading ? (
            <div className="p-10 flex items-center justify-center gap-2 text-ink-muted">
              <Loader2 className="h-5 w-5 animate-spin" />
              Loading trained phrases…
            </div>
          ) : phrases.length === 0 ? (
            <div className="p-14 flex flex-col items-center text-center gap-3">
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-600">
                <BookOpen className="h-6 w-6" strokeWidth={2} />
              </span>
              <p className="text-ink font-semibold">No trained phrases yet</p>
              <p className="text-sm text-ink-muted max-w-sm">
                Add your first AI-sounding phrase and how you&rsquo;d rewrite it — the
                humanizer will use it every time it sees that pattern again.
              </p>
              <button
                onClick={() => setModal({ mode: "create" })}
                className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2.5 shadow-lift hover:brightness-110 transition mt-2"
              >
                <Plus className="h-4 w-4" strokeWidth={2.5} />
                Add Phrase
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-surface-muted text-left text-xs font-semibold uppercase tracking-wider text-ink-faint">
                    <th className="px-5 py-3 font-semibold w-[38%]">AI-sounding phrase</th>
                    <th className="px-5 py-3 font-semibold w-[38%]">Your humanized version</th>
                    <th className="px-5 py-3 font-semibold whitespace-nowrap">Added</th>
                    <th className="px-5 py-3 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {phrases.map((p) => (
                    <tr key={p.id} className="hover:bg-surface-muted/60 transition-colors align-top">
                      <td className="px-5 py-4 text-ink-muted italic">
                        <p className="line-clamp-3">&ldquo;{p.ai_phrase}&rdquo;</p>
                      </td>
                      <td className="px-5 py-4 text-ink font-medium">
                        <p className="line-clamp-3">{p.humanized_phrase}</p>
                      </td>
                      <td className="px-5 py-4 text-ink-faint whitespace-nowrap">
                        {formatDate(p.created_at)}
                      </td>
                      <td className="px-5 py-4">
                        <div className="flex items-center justify-end gap-1.5">
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
                                className="rounded-md p-1.5 text-ink-muted hover:text-ink hover:bg-surface transition-colors"
                                title="Cancel"
                              >
                                <X className="h-3.5 w-3.5" />
                              </button>
                            </>
                          ) : (
                            <>
                              <button
                                onClick={() => setModal({ mode: "edit", phrase: p })}
                                className="rounded-md p-1.5 text-ink-muted hover:text-brand-600 hover:bg-brand-50 transition-colors"
                                title="Edit"
                              >
                                <Pencil className="h-3.5 w-3.5" />
                              </button>
                              <button
                                onClick={() => setConfirmingId(p.id)}
                                className="rounded-md p-1.5 text-ink-muted hover:text-danger-600 hover:bg-danger-50 transition-colors"
                                title="Delete"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <PhraseFormModal state={modal} onClose={() => setModal(null)} onSave={handleSave} />
    </main>
  );
}

function PhraseFormModal({
  state,
  onClose,
  onSave,
}: {
  state: ModalState;
  onClose: () => void;
  onSave: (aiPhrase: string, humanizedPhrase: string) => Promise<void>;
}) {
  const [aiPhrase, setAiPhrase] = useState("");
  const [humanizedPhrase, setHumanizedPhrase] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (state?.mode === "edit") {
      setAiPhrase(state.phrase.ai_phrase);
      setHumanizedPhrase(state.phrase.humanized_phrase);
    } else {
      setAiPhrase("");
      setHumanizedPhrase("");
    }
    setError(null);
  }, [state]);

  async function handleSubmit() {
    if (!aiPhrase.trim() || !humanizedPhrase.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await onSave(aiPhrase, humanizedPhrase);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal
      open={state !== null}
      onClose={onClose}
      title={state?.mode === "edit" ? "Edit Trained Phrase" : "Add Trained Phrase"}
    >
      <div className="space-y-4">
        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-1.5 block">
            AI-sounding phrase
          </label>
          <textarea
            value={aiPhrase}
            onChange={(e) => setAiPhrase(e.target.value)}
            placeholder="I like learning new things and trying to improve myself little by little."
            rows={3}
            className="w-full rounded-lg border border-border bg-surface-muted focus:border-brand-400 focus:bg-surface outline-none px-3 py-2.5 text-sm leading-relaxed placeholder:text-ink-faint resize-y transition-colors"
          />
        </div>
        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-1.5 block">
            Your humanized version
          </label>
          <textarea
            value={humanizedPhrase}
            onChange={(e) => setHumanizedPhrase(e.target.value)}
            placeholder="I enjoy picking up new things and getting a little better at them over time."
            rows={3}
            className="w-full rounded-lg border border-border bg-surface-muted focus:border-brand-400 focus:bg-surface outline-none px-3 py-2.5 text-sm leading-relaxed placeholder:text-ink-faint resize-y transition-colors"
          />
        </div>

        {error && (
          <p className="flex items-center gap-2 text-sm text-danger-600 bg-danger-50 border border-danger-400/30 rounded-lg px-3 py-2.5">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        <div className="flex justify-end gap-2 pt-1">
          <button
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm font-medium text-ink-muted hover:bg-surface-muted transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={saving || !aiPhrase.trim() || !humanizedPhrase.trim()}
            className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2 shadow-lift disabled:opacity-50 disabled:cursor-not-allowed hover:brightness-110 transition"
          >
            {saving && <Loader2 className="h-4 w-4 animate-spin" />}
            {state?.mode === "edit" ? "Save Changes" : "Add Phrase"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
