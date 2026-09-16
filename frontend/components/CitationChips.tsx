import type { Citation } from "@/lib/types";

function fileName(source: string): string {
  return source.split("/").pop() ?? source;
}

export function CitationChips({ citations }: { citations: Citation[] }) {
  if (!citations.length) return null;

  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {citations.map((citation, index) => (
        <span
          key={`${citation.source}-${citation.page}-${index}`}
          title={citation.source}
          className="rounded border border-border bg-background px-1.5 py-0.5 font-mono text-xs text-muted"
        >
          {fileName(citation.source)}
          {citation.page != null ? ` · p.${citation.page}` : ""}
        </span>
      ))}
    </div>
  );
}
