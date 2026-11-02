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

  if (isEmpty) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center px-4 text-center">
        <h2 className="text-3xl font-semibold tracking-tight text-foreground sm:text-4xl">
          ISO 15189 QMS Assistant
        </h2>
        <p className="mt-3 max-w-md text-sm text-muted">
          Ask a question about the ISO 15189:2022 standard to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-4 overflow-y-auto px-4 py-6">
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}
      {isStreaming && <StreamingBubble text={streamingText} citations={citations} />}
      <div ref={bottomRef} />
    </div>
  );
}
