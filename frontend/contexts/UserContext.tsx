"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { User, listUsers, createUser as apiCreateUser, deleteUser as apiDeleteUser } from "@/lib/api";

const STORAGE_KEY = "contentiq_active_user_id";

interface UserContextValue {
  users: User[];
  activeUserId: number | undefined;
  activeUser: User | undefined;
  loading: boolean;
  setActiveUserId: (id: number) => void;
  createUser: (name: string) => Promise<User>;
  deleteUser: (id: number) => Promise<void>;
}

const UserContext = createContext<UserContextValue | null>(null);

export function UserProvider({ children }: { children: ReactNode }) {
  const [users, setUsers] = useState<User[]>([]);
  const [activeUserId, setActiveUserIdState] = useState<number | undefined>(undefined);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listUsers()
      .then((res) => {
        setUsers(res);
        let stored: number | undefined;
        try {
          const raw = window.localStorage.getItem(STORAGE_KEY);
          stored = raw ? Number(raw) : undefined;
        } catch {
          stored = undefined;
        }
        const validStored = stored !== undefined && res.some((u) => u.id === stored) ? stored : undefined;
        setActiveUserIdState(validStored ?? res[0]?.id);
      })
      .catch(() => {
        // Degrade gracefully — the backend falls back to a default user
        // internally, so the app stays usable even if this fails.
      })
      .finally(() => setLoading(false));
  }, []);

  function setActiveUserId(id: number) {
    setActiveUserIdState(id);
    try {
      window.localStorage.setItem(STORAGE_KEY, String(id));
    } catch {
      // ignore storage errors (private browsing, storage disabled, etc.)
    }
  }

  async function createUser(name: string): Promise<User> {
    const created = await apiCreateUser(name);
    setUsers((prev) => [...prev, created]);
    setActiveUserId(created.id);
    return created;
  }

  async function deleteUser(id: number): Promise<void> {
    await apiDeleteUser(id);
    const remaining = users.filter((u) => u.id !== id);
    setUsers(remaining);
    if (activeUserId === id) {
      setActiveUserId(remaining[0]?.id as number);
    }
  }

  const activeUser = users.find((u) => u.id === activeUserId);

  return (
    <UserContext.Provider
      value={{ users, activeUserId, activeUser, loading, setActiveUserId, createUser, deleteUser }}
    >
      {children}
    </UserContext.Provider>
  );
}

export function useUser(): UserContextValue {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error("useUser must be used within a UserProvider");
  return ctx;
}
