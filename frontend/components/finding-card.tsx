import type { Finding } from "@/lib/types";

type Props = {
  finding: Finding;
  note?: string;
  // Given only while the review waits for approval: shows a "post it" checkbox.
  selected?: boolean;
  onToggle?: () => void;
};

export function FindingCard({ finding, note, selected, onToggle }: Props) {
  return (
    <article className={`card${selected === false ? " dismissed" : ""}`}>
      <header className="row">
        {onToggle && (
          <label className="row">
            <input type="checkbox" checked={selected} onChange={onToggle} /> post
          </label>
        )}
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
