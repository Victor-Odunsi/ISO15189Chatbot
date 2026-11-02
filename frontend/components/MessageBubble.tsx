import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import type { Message } from "@/lib/types";
import { CitationChips } from "./CitationChips";
import { ErrorBanner } from "./ErrorBanner";

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  return (
    <div className={clsx("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={clsx(
          "max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
          isUser ? "bg-accent text-accent-foreground" : "border border-border bg-surface text-foreground"
        )}
      >
        {message.error ? (
          <ErrorBanner message={message.error} />
        ) : (
          <div className="markdown-content">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        {!isUser && message.citations && <CitationChips citations={message.citations} />}
      </div>
    </div>
  );
}
