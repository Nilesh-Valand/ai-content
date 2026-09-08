"use client";

import { useRef, useState } from "react";
import { ChevronDown, Check, User as UserIcon, Plus, Loader2, AlertCircle } from "lucide-react";
import { useDismiss } from "@/hooks/useDismiss";
import { useUser } from "@/contexts/UserContext";

export default function UserSwitcher() {
  const { users, activeUser, activeUserId, setActiveUserId, createUser } = useUser();
  const [open, setOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ref = useRef<HTMLDivElement>(null);
  function close() {
    setOpen(false);
    setCreating(false);
    setNewName("");
    setError(null);
  }
  useDismiss(ref, open, close);

  async function handleCreate() {
    if (!newName.trim()) return;
    setSaving(true);
    setError(null);
    try {
      await createUser(newName);
      close();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not create user.");
    } finally {
      setSaving(false);
    }
  }

  if (users.length === 0) return null;

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => (open ? close() : setOpen(true))}
        className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface-muted px-3 py-1.5 text-sm font-medium text-ink hover:border-border-strong transition-colors"
      >
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-brand-100 text-brand-700 text-[11px] font-bold shrink-0">
          {(activeUser?.name ?? "?").slice(0, 1).toUpperCase()}
        </span>
        {activeUser?.name ?? "Select user"}
        <ChevronDown className={`h-3.5 w-3.5 text-ink-faint transition-transform ${open ? "rotate-180" : ""}`} />
      </button>

      {open && (
        <div className="fade-in absolute right-0 z-30 mt-2 w-64 overflow-hidden rounded-xl border border-border bg-surface shadow-card">
          <ul className="max-h-60 overflow-y-auto py-1">
            {users.map((u) => {
              const active = u.id === activeUserId;
              return (
                <li key={u.id}>
                  <button
                    onClick={() => {
                      setActiveUserId(u.id);
                      close();
                    }}
                    className={`flex w-full items-center justify-between gap-2 px-3.5 py-2.5 text-left text-sm transition-colors ${
                      active ? "bg-brand-50 text-brand-700" : "text-ink hover:bg-surface-muted"
                    }`}
                  >
                    <span className="flex items-center gap-2 min-w-0">
                      <UserIcon className={`h-3.5 w-3.5 shrink-0 ${active ? "text-brand-600" : "text-ink-faint"}`} />
                      <span className="truncate font-medium">{u.name}</span>
                    </span>
                    {active && <Check className="h-3.5 w-3.5 shrink-0 text-brand-600" />}
                  </button>
                </li>
              );
            })}
          </ul>

          <div className="border-t border-border p-2.5">
            {creating ? (
              <div className="space-y-2">
                <input
                  autoFocus
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                  placeholder="New user's name"
                  className="w-full rounded-lg border border-border bg-surface-muted focus:border-brand-400 focus:bg-surface outline-none px-2.5 py-1.5 text-sm placeholder:text-ink-faint transition-colors"
                />
                {error && (
                  <p className="flex items-center gap-1.5 text-xs text-danger-600">
                    <AlertCircle className="h-3.5 w-3.5 shrink-0" />
                    {error}
                  </p>
                )}
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => {
                      setCreating(false);
                      setNewName("");
                      setError(null);
                    }}
                    className="rounded-md px-2.5 py-1 text-xs font-medium text-ink-muted hover:text-ink transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreate}
                    disabled={saving || !newName.trim()}
                    className="inline-flex items-center gap-1.5 rounded-md bg-brand-gradient text-white text-xs font-semibold px-3 py-1.5 disabled:opacity-50 hover:brightness-110 transition"
                  >
                    {saving && <Loader2 className="h-3 w-3 animate-spin" />}
                    Create
                  </button>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setCreating(true)}
                className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm font-medium text-brand-600 hover:bg-brand-50 transition-colors"
              >
                <Plus className="h-4 w-4" strokeWidth={2.5} />
                Add new user
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
