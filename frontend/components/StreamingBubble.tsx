import ReactMarkdown from "react-markdown";
import type { Citation } from "@/lib/types";
import { CitationChips } from "./CitationChips";

interface StreamingBubbleProps {
  text: string;
  citations: Citation[];
}

export function StreamingBubble({ text, citations }: StreamingBubbleProps) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl border border-border bg-surface px-4 py-2.5 text-sm leading-relaxed text-foreground">
        <div className="markdown-content">
          <ReactMarkdown>{text || "..."}</ReactMarkdown>
        </div>
        <CitationChips citations={citations} />
      </div>
    </div>
  );
}
