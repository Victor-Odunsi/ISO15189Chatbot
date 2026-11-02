"use client";

import { useCallback, useEffect, useState } from "react";
import { deleteSession as apiDeleteSession, getSessionMessages } from "@/lib/api";
import type { Message, SessionSummary } from "@/lib/types";

const SESSIONS_KEY = "iso15189.sessions";
const ACTIVE_SESSION_KEY = "iso15189.activeSessionId";

function readSessions(): SessionSummary[] {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY);
    return raw ? (JSON.parse(raw) as SessionSummary[]) : [];
  } catch {
    return [];
  }
}

function writeSessions(sessions: SessionSummary[]) {
  try {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  } catch {
    // localStorage unavailable (private mode, etc.) -- sessions just won't persist across reloads
  }
}

function titleFromMessage(content: string): string {
  const trimmed = content.trim();
  if (!trimmed) return "New chat";
  return trimmed.length > 50 ? `${trimmed.slice(0, 50)}...` : trimmed;
}

export function useSessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setSessions(readSessions());

    let active = localStorage.getItem(ACTIVE_SESSION_KEY);
    if (!active) {
      active = crypto.randomUUID();
      localStorage.setItem(ACTIVE_SESSION_KEY, active);
    }
    setActiveSessionId(active);
    setLoaded(true);
  }, []);

  useEffect(() => {
    if (!activeSessionId) return;
    let cancelled = false;
    getSessionMessages(activeSessionId).then((rows) => {
      if (cancelled) return;
      setMessages(rows.map((row) => ({ role: row.role, content: row.content })));
    });
    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  const touchSession = useCallback((sessionId: string, firstUserMessage?: string) => {
    setSessions((prev) => {
      const existing = prev.find((s) => s.id === sessionId);
      const next: SessionSummary = {
        id: sessionId,
        title: existing?.title ?? titleFromMessage(firstUserMessage ?? ""),
        updatedAt: new Date().toISOString(),
      };
      const updated = [next, ...prev.filter((s) => s.id !== sessionId)];
      writeSessions(updated);
      return updated;
    });
  }, []);

  const newChat = useCallback(() => {
    const id = crypto.randomUUID();
    localStorage.setItem(ACTIVE_SESSION_KEY, id);
    setActiveSessionId(id);
    setMessages([]);
  }, []);

  const switchSession = useCallback((sessionId: string) => {
    localStorage.setItem(ACTIVE_SESSION_KEY, sessionId);
    setActiveSessionId(sessionId);
  }, []);

  const removeSession = useCallback(
    async (sessionId: string) => {
      await apiDeleteSession(sessionId);
      setSessions((prev) => {
        const updated = prev.filter((s) => s.id !== sessionId);
        writeSessions(updated);
        return updated;
      });
      if (sessionId === activeSessionId) {
        newChat();
      }
    },
    [activeSessionId, newChat]
  );

  return {
    sessions,
    activeSessionId,
    messages,
    setMessages,
    loaded,
    touchSession,
    newChat,
    switchSession,
    removeSession,
  };
}
