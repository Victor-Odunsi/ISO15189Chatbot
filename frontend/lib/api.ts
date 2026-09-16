import type { StreamEvent } from "./types";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Streams Server-Sent Events from POST /chat. Uses fetch + ReadableStream
 * rather than EventSource, since EventSource can't send a POST body.
 */
export async function* streamChat(
  question: string,
  sessionId: string,
  signal: AbortSignal
): AsyncGenerator<StreamEvent> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Chat request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawFrame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const event = parseFrame(rawFrame);
      if (event) yield event;
      boundary = buffer.indexOf("\n\n");
    }
  }
}

function parseFrame(rawFrame: string): StreamEvent | null {
  let eventType: StreamEvent["type"] = "token";
  let dataLine = "";

  for (const line of rawFrame.split("\n")) {
    if (line.startsWith("event:")) {
      eventType = line.slice("event:".length).trim() as StreamEvent["type"];
    } else if (line.startsWith("data:")) {
      dataLine += line.slice("data:".length).trim();
    }
  }

  if (!dataLine) return null;

  try {
    const payload = JSON.parse(dataLine);
    return { type: eventType, ...payload };
  } catch {
    return null;
  }
}

export interface HistoryRow {
  role: "user" | "assistant";
  content: string;
}

export async function getSessionMessages(sessionId: string): Promise<HistoryRow[]> {
  try {
    const response = await fetch(`${API_URL}/sessions/${sessionId}/messages`);
    if (!response.ok) return [];
    return await response.json();
  } catch {
    return [];
  }
}

export async function deleteSession(sessionId: string): Promise<void> {
  try {
    await fetch(`${API_URL}/sessions/${sessionId}`, { method: "DELETE" });
  } catch {
    // best-effort -- the local sidebar entry is removed regardless
  }
}
