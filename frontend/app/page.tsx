"use client";

import { useCallback, useState } from "react";
import { ChatInput } from "@/components/ChatInput";
import { ChatWindow } from "@/components/ChatWindow";
import { ErrorBanner } from "@/components/ErrorBanner";
import { Sidebar } from "@/components/Sidebar";
import { useChatStream } from "@/hooks/useChatStream";
import { useSessions } from "@/hooks/useSessions";
import type { Message } from "@/lib/types";

export default function Home() {
  const {
    sessions,
    activeSessionId,
    messages,
    setMessages,
    loaded,
    touchSession,
    newChat,
    switchSession,
    removeSession,
  } = useSessions();
  const { status, streamingText, citations, send } = useChatStream();
  const [bannerError, setBannerError] = useState<string | null>(null);

  const handleSend = useCallback(
    async (question: string) => {
      if (!activeSessionId) return;
      setBannerError(null);

      const userMessage: Message = { role: "user", content: question };
      setMessages((prev) => [...prev, userMessage]);
      touchSession(activeSessionId, question);

      const result = await send(question, activeSessionId);

      if (result.ok) {
        setMessages((prev) => [...prev, { role: "assistant", content: result.answer, citations: result.citations }]);
      } else {
        setBannerError(result.error);
      }
    },
    [activeSessionId, send, setMessages, touchSession]
  );

  if (!loaded) return null;

  return (
    <div className="flex h-screen">
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={newChat}
        onSwitch={switchSession}
        onDelete={removeSession}
      />
      <div className="flex flex-1 flex-col">
        <header className="border-b border-border bg-surface px-4 py-3">
          <h1 className="text-sm font-semibold text-foreground">ISO 15189 QMS Assistant</h1>
        </header>
        <ChatWindow
          messages={messages}
          isStreaming={status === "streaming"}
          streamingText={streamingText}
          citations={citations}
        />
        {bannerError && (
          <div className="px-4 pb-2">
            <ErrorBanner message={bannerError} />
          </div>
        )}
        <ChatInput disabled={status === "streaming"} onSend={handleSend} />
      </div>
    </div>
  );
}
