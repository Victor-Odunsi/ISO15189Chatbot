"use client";

import { useCallback, useState } from "react";
import { streamChat } from "@/lib/api";
import type { Citation } from "@/lib/types";

// A cold Lambda container can legitimately take a while before the first
// byte (backend's own function timeout is 120s precisely to absorb this),
// so 30s was too tight and aborted genuinely-working requests, not just
// stuck ones.
const IDLE_TIMEOUT_MS = 90_000;

export type ChatStreamStatus = "idle" | "streaming" | "error";

type SendResult =
  | { ok: true; answer: string; citations: Citation[] }
  | { ok: false; error: string };

interface UseChatStreamResult {
  status: ChatStreamStatus;
  streamingText: string;
  citations: Citation[];
  send: (question: string, sessionId: string) => Promise<SendResult>;
}

export function useChatStream(): UseChatStreamResult {
  const [status, setStatus] = useState<ChatStreamStatus>("idle");
  const [streamingText, setStreamingText] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);

  const send = useCallback(async (question: string, sessionId: string): Promise<SendResult> => {
    setStatus("streaming");
    setStreamingText("");
    setCitations([]);

    const controller = new AbortController();
    let idleTimer: ReturnType<typeof setTimeout> = setTimeout(() => controller.abort(), IDLE_TIMEOUT_MS);
    const resetIdleTimer = () => {
      clearTimeout(idleTimer);
      idleTimer = setTimeout(() => controller.abort(), IDLE_TIMEOUT_MS);
    };

    let answer = "";
    let finalCitations: Citation[] = [];

    try {
      for await (const event of streamChat(question, sessionId, controller.signal)) {
        resetIdleTimer();

        if (event.type === "token" && event.content) {
          answer += event.content;
          setStreamingText(answer);
        } else if (event.type === "citations" && event.citations) {
          finalCitations = event.citations;
          setCitations(event.citations);
        } else if (event.type === "error") {
          const message = event.message ?? "Something went wrong.";
          setStatus("error");
          return { ok: false, error: message };
        } else if (event.type === "end") {
          break;
        }
      }
    } catch {
      const message = controller.signal.aborted
        ? "The response timed out. Please try again."
        : "Connection lost. Please try again.";
      setStatus("error");
      return { ok: false, error: message };
    } finally {
      clearTimeout(idleTimer);
    }

    setStatus("idle");
    return { ok: true, answer, citations: finalCitations };
  }, []);

  return { status, streamingText, citations, send };
}
