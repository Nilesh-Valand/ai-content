"use client";

import { useEffect, useState } from "react";
import {
  Profile,
  TrainedPhrase,
  listProfiles,
  createProfile,
  updateProfile,
  deleteProfile,
  listTrainedPhrases,
  createTrainedPhrase,
  updateTrainedPhrase,
  deleteTrainedPhrase,
} from "@/lib/api";
import { useUser } from "@/contexts/UserContext";
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
  FolderPlus,
  Sparkles,
} from "lucide-react";

type PhraseModalState = { mode: "create" } | { mode: "edit"; phrase: TrainedPhrase } | null;
type ProfileModalState = { mode: "create" } | { mode: "edit"; profile: Profile } | null;

function formatDate(iso: string) {
  const d = new Date(iso.replace(" ", "T") + "Z");
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export default function TrainPage() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<number | null>(null);
  const [phrases, setPhrases] = useState<TrainedPhrase[]>([]);
  
  const [loadingProfiles, setLoadingProfiles] = useState(true);
  const [loadingPhrases, setLoadingPhrases] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [phraseModal, setPhraseModal] = useState<PhraseModalState>(null);
  const [profileModal, setProfileModal] = useState<ProfileModalState>(null);
  
  const [confirmingPhraseId, setConfirmingPhraseId] = useState<number | null>(null);
  const [deletingPhraseId, setDeletingPhraseId] = useState<number | null>(null);

  const [confirmingDeleteProfileId, setConfirmingDeleteProfileId] = useState<number | null>(null);
  const [deletingProfileId, setDeletingProfileId] = useState<number | null>(null);

  const { activeUserId } = useUser();

  useEffect(() => {
    if (activeUserId === undefined) return;
    loadInitialProfiles();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeUserId]);

  useEffect(() => {
    if (activeProfileId !== null) {
      loadPhrases(activeProfileId);
    }
  }, [activeProfileId]);

  async function loadInitialProfiles() {
    setLoadingProfiles(true);
    setError(null);
    try {
      const res = await listProfiles(activeUserId);
      setProfiles(res);
      setActiveProfileId(res[0]?.id ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load phrase profiles.");
    } finally {
      setLoadingProfiles(false);
    }
  }

  async function loadPhrases(profileId: number) {
    setLoadingPhrases(true);
    setError(null);
    try {
      const res = await listTrainedPhrases(profileId);
      setPhrases(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load phrases.");
    } finally {
      setLoadingPhrases(false);
    }
  }

  const activeProfile = profiles.find((p) => p.id === activeProfileId) || profiles[0];

  async function handleSaveProfile(name: string, description: string) {
    if (profileModal?.mode === "edit") {
      const updated = await updateProfile(profileModal.profile.id, name, description);
      setProfiles((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
    } else {
      const created = await createProfile(name, description, activeUserId);
      setProfiles((prev) => [...prev, created]);
      setActiveProfileId(created.id);
    }
    setProfileModal(null);
  }

  async function handleDeleteProfile(id: number) {
    setDeletingProfileId(id);
    setError(null);
    try {
      await deleteProfile(id);
      const updatedProfiles = profiles.filter((p) => p.id !== id);
      setProfiles(updatedProfiles);
      if (activeProfileId === id) {
        setActiveProfileId(updatedProfiles[0]?.id ?? null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete profile.");
    } finally {
      setDeletingProfileId(null);
      setConfirmingDeleteProfileId(null);
    }
  }

  async function handleSavePhrase(aiPhrase: string, humanizedPhrase: string) {
    if (!activeProfileId) return;

    if (phraseModal?.mode === "edit") {
      const updated = await updateTrainedPhrase(
        phraseModal.phrase.id,
        aiPhrase,
        humanizedPhrase,
        activeProfileId
      );
      setPhrases((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
    } else {
      const created = await createTrainedPhrase(aiPhrase, humanizedPhrase, activeProfileId);
      setPhrases((prev) => [created, ...prev]);
      // Update profile count locally
      setProfiles((prev) =>
        prev.map((p) => (p.id === activeProfileId ? { ...p, phrase_count: p.phrase_count + 1 } : p))
      );
    }
    setPhraseModal(null);
  }

  async function handleDeletePhrase(id: number) {
    setDeletingPhraseId(id);
    setError(null);
    try {
      await deleteTrainedPhrase(id);
      setPhrases((prev) => prev.filter((p) => p.id !== id));
      // Update profile count locally
      if (activeProfileId) {
        setProfiles((prev) =>
          prev.map((p) =>
            p.id === activeProfileId ? { ...p, phrase_count: Math.max(0, p.phrase_count - 1) } : p
          )
        );
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not delete phrase.");
    } finally {
      setDeletingPhraseId(null);
      setConfirmingPhraseId(null);
    }
  }

  return (
    <main className="w-full bg-hero-gradient bg-no-repeat">
      <div className="w-full px-4 sm:px-8 lg:px-12 py-10 lg:py-14">
        {/* Page Header */}
        <div className="flex flex-wrap items-start justify-between gap-4 mb-8">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-1.5 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 mb-4">
              <GraduationCap className="h-3.5 w-3.5" strokeWidth={2.5} />
              Personal Style Training
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-ink">
              Phrase Profiles
            </h1>
            <p className="text-ink-muted mt-3 text-[15px] leading-relaxed">
              Create writing style profiles for different target voices (e.g. LinkedIn, Tech Blog, Academic).
              Add phrase examples under each profile to train the AI humanizer for that exact style.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setProfileModal({ mode: "create" })}
              className="inline-flex items-center gap-2 rounded-lg border border-border bg-surface text-ink text-sm font-semibold px-3.5 py-2.5 shadow-sm hover:bg-surface-muted transition"
            >
              <FolderPlus className="h-4 w-4 text-brand-600" strokeWidth={2} />
              New Profile
            </button>
            <button
              onClick={() => setPhraseModal({ mode: "create" })}
              disabled={!activeProfileId}
              className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2.5 shadow-lift hover:brightness-110 transition shrink-0 disabled:opacity-50"
            >
              <Plus className="h-4 w-4" strokeWidth={2.5} />
              Add Phrase
            </button>
          </div>
        </div>

        {error && (
          <p className="flex items-center gap-2 text-sm text-danger-600 bg-danger-50 border border-danger-400/30 rounded-xl px-4 py-3 mb-6">
            <AlertCircle className="h-4 w-4 shrink-0" />
            {error}
          </p>
        )}

        {/* Profiles Navigation Tabs */}
        {loadingProfiles ? (
          <div className="flex items-center gap-2 text-sm text-ink-muted py-4">
            <Loader2 className="h-4 w-4 animate-spin" /> Loading style profiles…
          </div>
        ) : profiles.length === 0 ? (
          <div className="mb-6 rounded-xl border border-dashed border-border px-4 py-3 text-sm text-ink-muted">
            No style profiles yet — create one with &ldquo;New Profile&rdquo; to start training phrases.
          </div>
        ) : (
          <div className="mb-6 overflow-x-auto pb-2">
            <div className="flex items-center gap-2 border-b border-border pb-3">
              {profiles.map((p) => {
                const isActive = p.id === activeProfileId;
                return (
                  <div key={p.id} className="relative group flex items-center">
                    <button
                      onClick={() => setActiveProfileId(p.id)}
                      className={`inline-flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all ${
                        isActive
                          ? "bg-brand-600 text-white shadow-lift"
                          : "bg-surface border border-border text-ink-muted hover:text-ink hover:bg-surface-muted"
                      }`}
                    >
                      <Sparkles className={`h-3.5 w-3.5 ${isActive ? "text-white" : "text-brand-500"}`} />
                      <span>{p.name}</span>
                      <span
                        className={`rounded-full text-[11px] font-semibold px-2 py-0.5 ${
                          isActive
                            ? "bg-white/20 text-white"
                            : "bg-surface-muted text-ink-faint border border-border"
                        }`}
                      >
                        {p.phrase_count}
                      </span>
                    </button>

                    {/* Profile edit/delete actions */}
                    {isActive && (
                      <div className="flex items-center gap-1 ml-1.5 bg-surface border border-border rounded-lg p-1">
                        <button
                          onClick={() => setProfileModal({ mode: "edit", profile: p })}
                          className="p-1 text-ink-muted hover:text-brand-600 hover:bg-brand-50 rounded transition"
                          title="Edit Profile Name"
                        >
                          <Pencil className="h-3 w-3" />
                        </button>
                        {confirmingDeleteProfileId === p.id ? (
                          <div className="flex items-center gap-1">
                            <button
                              onClick={() => handleDeleteProfile(p.id)}
                              disabled={deletingProfileId === p.id}
                              className="p-1 text-white bg-danger-500 hover:bg-danger-600 rounded transition"
                              title="Confirm delete profile"
                            >
                              {deletingProfileId === p.id ? (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              ) : (
                                <Check className="h-3 w-3" />
                              )}
                            </button>
                            <button
                              onClick={() => setConfirmingDeleteProfileId(null)}
                              className="p-1 text-ink-muted hover:text-ink rounded transition"
                            >
                              <X className="h-3 w-3" />
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => setConfirmingDeleteProfileId(p.id)}
                            className="p-1 text-ink-muted hover:text-danger-600 hover:bg-danger-50 rounded transition"
                            title="Delete Profile"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {activeProfile?.description && (
              <p className="text-xs text-ink-muted mt-2 italic px-1">
                {activeProfile.description}
              </p>
            )}
          </div>
        )}

        {/* Phrases Table */}
        <div className="rounded-2xl border border-border bg-surface shadow-card overflow-hidden">
          {loadingPhrases ? (
            <div className="p-10 flex items-center justify-center gap-2 text-ink-muted">
              <Loader2 className="h-5 w-5 animate-spin" />
              Loading trained phrases for {activeProfile?.name || "profile"}…
            </div>
          ) : phrases.length === 0 ? (
            <div className="p-14 flex flex-col items-center text-center gap-3">
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-brand-50 text-brand-600">
                <BookOpen className="h-6 w-6" strokeWidth={2} />
              </span>
              {profiles.length === 0 ? (
                <>
                  <p className="text-ink font-semibold">No style profile yet</p>
                  <p className="text-sm text-ink-muted max-w-sm">
                    Create one with &ldquo;New Profile&rdquo; above, then add trained phrases to it.
                  </p>
                </>
              ) : (
                <>
                  <p className="text-ink font-semibold">
                    No trained phrases in &ldquo;{activeProfile?.name}&rdquo; yet
                  </p>
                  <p className="text-sm text-ink-muted max-w-sm">
                    Add AI-sounding phrases and how you rewrite them for this style profile. The AI humanizer will learn this style.
                  </p>
                  <button
                    onClick={() => setPhraseModal({ mode: "create" })}
                    className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2.5 shadow-lift hover:brightness-110 transition mt-2"
                  >
                    <Plus className="h-4 w-4" strokeWidth={2.5} />
                    Add Phrase to Profile
                  </button>
                </>
              )}
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
                          {confirmingPhraseId === p.id ? (
                            <>
                              <span className="text-xs text-ink-muted mr-1">Delete?</span>
                              <button
                                onClick={() => handleDeletePhrase(p.id)}
                                disabled={deletingPhraseId === p.id}
                                className="rounded-md p-1.5 text-white bg-danger-500 hover:bg-danger-600 transition-colors disabled:opacity-50"
                                title="Confirm delete"
                              >
                                {deletingPhraseId === p.id ? (
                                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                ) : (
                                  <Check className="h-3.5 w-3.5" />
                                )}
                              </button>
                              <button
                                onClick={() => setConfirmingPhraseId(null)}
                                className="rounded-md p-1.5 text-ink-muted hover:text-ink hover:bg-surface transition-colors"
                                title="Cancel"
                              >
                                <X className="h-3.5 w-3.5" />
                              </button>
                            </>
                          ) : (
                            <>
                              <button
                                onClick={() => setPhraseModal({ mode: "edit", phrase: p })}
                                className="rounded-md p-1.5 text-ink-muted hover:text-brand-600 hover:bg-brand-50 transition-colors"
                                title="Edit"
                              >
                                <Pencil className="h-3.5 w-3.5" />
                              </button>
                              <button
                                onClick={() => setConfirmingPhraseId(p.id)}
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

      {/* Phrase Modal */}
      <PhraseFormModal
        state={phraseModal}
        profileName={activeProfile?.name || ""}
        onClose={() => setPhraseModal(null)}
        onSave={handleSavePhrase}
      />

      {/* Profile Modal */}
      <ProfileFormModal
        state={profileModal}
        onClose={() => setProfileModal(null)}
        onSave={handleSaveProfile}
      />
    </main>
  );
}

function ProfileFormModal({
  state,
  onClose,
  onSave,
}: {
  state: ProfileModalState;
  onClose: () => void;
  onSave: (name: string, description: string) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (state?.mode === "edit") {
      setName(state.profile.name);
      setDescription(state.profile.description || "");
    } else {
      setName("");
      setDescription("");
    }
    setError(null);
  }, [state]);

  async function handleSubmit() {
    if (!name.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await onSave(name, description);
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
      title={state?.mode === "edit" ? "Edit Phrase Profile" : "Create New Phrase Profile"}
    >
      <div className="space-y-4">
        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-1.5 block">
            Profile Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. LinkedIn Casual Voice, Tech Blog, Academic Writing"
            className="w-full rounded-lg border border-border bg-surface-muted focus:border-brand-400 focus:bg-surface outline-none px-3 py-2.5 text-sm leading-relaxed placeholder:text-ink-faint transition-colors"
          />
        </div>

        <div>
          <label className="text-xs font-semibold uppercase tracking-wider text-ink-faint mb-1.5 block">
            Description (Optional)
          </label>
          <textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Brief description of where to use this writing style profile..."
            rows={2}
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
            disabled={saving || !name.trim()}
            className="inline-flex items-center gap-2 rounded-lg bg-brand-gradient text-white text-sm font-semibold px-4 py-2 shadow-lift disabled:opacity-50 disabled:cursor-not-allowed hover:brightness-110 transition"
          >
            {saving && <Loader2 className="h-4 w-4 animate-spin" />}
            {state?.mode === "edit" ? "Save Changes" : "Create Profile"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

function PhraseFormModal({
  state,
  profileName,
  onClose,
  onSave,
}: {
  state: PhraseModalState;
  profileName: string;
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
      title={
        state?.mode === "edit"
          ? `Edit Phrase (${profileName})`
          : `Add Trained Phrase to "${profileName}"`
      }
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
            Your humanized version for this profile
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

