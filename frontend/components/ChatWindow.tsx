"use client";

import { useEffect, useRef } from "react";
import type { Citation, Message } from "@/lib/types";
import { MessageBubble } from "./MessageBubble";
import { StreamingBubble } from "./StreamingBubble";

interface ChatWindowProps {
  messages: Message[];
  isStreaming: boolean;
  streamingText: string;
  citations: Citation[];
}

export function ChatWindow({ messages, isStreaming, streamingText, citations }: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingText]);

  const isEmpty = messages.length === 0 && !isStreaming;

  return (
    <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6">
      {isEmpty && (
        <p className="mx-auto max-w-md pt-16 text-center text-sm text-muted">
          Ask a question about the ISO 15189:2022 standard to get started.
        </p>
      )}
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}
      {isStreaming && <StreamingBubble text={streamingText} citations={citations} />}
      <div ref={bottomRef} />
    </div>
  );
}
