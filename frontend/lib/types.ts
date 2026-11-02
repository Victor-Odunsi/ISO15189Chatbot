export type Role = "user" | "assistant";

export interface Citation {
  source: string;
  page: number | null;
}

export interface Message {
  role: Role;
  content: string;
  citations?: Citation[];
  error?: string;
}

export interface SessionSummary {
  id: string;
  title: string;
  updatedAt: string;
}

export type StreamEventType = "session" | "citations" | "token" | "end" | "error";

export interface StreamEvent {
  type: StreamEventType;
  session_id?: string;
  citations?: Citation[];
  content?: string;
  message?: string;
}
