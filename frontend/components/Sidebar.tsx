"use client";

import clsx from "clsx";
import type { SessionSummary } from "@/lib/types";

interface SidebarProps {
  sessions: SessionSummary[];
  activeSessionId: string | null;
  onNewChat: () => void;
  onSwitch: (sessionId: string) => void;
  onDelete: (sessionId: string) => void;
}

export function Sidebar({ sessions, activeSessionId, onNewChat, onSwitch, onDelete }: SidebarProps) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-gradient-to-b from-surface to-background">
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="w-full rounded-lg border border-border px-3 py-2 text-left text-sm font-medium text-foreground hover:bg-background"
        >
          + New chat
        </button>
      </div>
      <nav className="flex-1 space-y-1 overflow-y-auto px-2 pb-3">
        {sessions.length === 0 && <p className="px-2 py-4 text-xs text-muted">No conversations yet.</p>}
        {sessions.map((session) => (
          <div
            key={session.id}
            className={clsx(
              "group flex items-center justify-between rounded-lg px-2 py-2 text-sm",
              session.id === activeSessionId ? "bg-background font-medium text-foreground" : "text-muted hover:bg-background"
            )}
          >
            <button onClick={() => onSwitch(session.id)} className="flex-1 truncate text-left">
              {session.title}
            </button>
            <button
              onClick={() => onDelete(session.id)}
              aria-label="Delete conversation"
              className="ml-2 hidden text-xs text-muted hover:text-error group-hover:inline"
            >
              ✕
            </button>
          </div>
        ))}
      </nav>
    </aside>
  );
}
