import type { Finding } from "@/lib/types";

export function FindingCard({ finding, note }: { finding: Finding; note?: string }) {
  return (
    <article className="card">
      <header className="row">
        <span className={`badge ${finding.severity}`}>{finding.severity}</span>
        <span className="muted">{finding.category}</span>
        <code className="muted">
          {finding.file}:{finding.line}
        </code>
      </header>
      <h3>{finding.title}</h3>
      <p>{finding.explanation}</p>
      <pre>{finding.suggestion}</pre>
      {note && <p className="muted">{note}</p>}
    </article>
  );
}
