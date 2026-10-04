"use client";

import Link from "next/link";
import { useParams } from "next/navigation";

import { FindingCard } from "@/components/finding-card";
import { SEVERITY_RANK, type Finding } from "@/lib/types";
import { useReviewEvents } from "@/lib/use-review-events";

// /reviews/<id>: the folder name [id] makes that part of the URL a parameter.
export default function ReviewPage() {
  const { id } = useParams<{ id: string }>();
  const review = useReviewEvents(id);
  const { pr, plan, agents, checks } = review;

  const kept = checks?.filter((c) => c.kept) ?? [];
  const dropped = checks?.filter((c) => !c.kept) ?? [];

  return (
    <>
      <p>
        <Link href="/">← new review</Link>
      </p>
      <h1>{pr ? `#${pr.number} ${pr.title}` : "Loading pull request…"}</h1>
      <p className="row">
        <span className={`badge status-${review.status}`}>{review.status}</span>
        {pr && (
          <span className="muted">
            {pr.files.length} files reviewed
            {pr.skipped.length > 0 && `, ${pr.skipped.length} skipped (${pr.skipped.join(", ")})`}
          </span>
        )}
      </p>
      {review.error && <p className="error">{review.error}</p>}

      {plan && (
        <section>
          <h2>Agents</h2>
          <p className="muted">Triage: {plan.reason}</p>
          <div className="grid">
            {Object.entries(agents).map(([name, agent]) => (
              <article key={name} className="card">
                <header className="row">
                  <strong>{name}</strong>
                  <span className="muted">
                    {agent.status === "working" ? "working…" : `done · ${agent.submitted} findings`}
                  </span>
                </header>
                <ul>
                  {agent.tools.map((line, i) => (
                    <li key={i}>
                      <code>{line}</code>
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </section>
      )}

      {checks ? (
        <section>
          <h2>Findings ({kept.length} verified)</h2>
          {kept.length === 0 && <p className="muted">No problems found.</p>}
          {bySeverity(kept.map((c) => c.finding)).map((f, i) => (
            <FindingCard key={i} finding={f} />
          ))}
          {dropped.length > 0 && (
            <details>
              <summary>{dropped.length} dropped by the verifier</summary>
              {dropped.map((c, i) => (
                <FindingCard key={i} finding={c.finding} note={`Dropped at ${c.stage}: ${c.reason}`} />
              ))}
            </details>
          )}
        </section>
      ) : (
        review.rawFindings.length > 0 && (
          <section>
            <h2>Findings so far (not verified yet)</h2>
            {review.rawFindings.map((f, i) => (
              <FindingCard key={i} finding={f} />
            ))}
          </section>
        )
      )}
    </>
  );
}

function bySeverity(findings: Finding[]): Finding[] {
  return [...findings].sort((a, b) => SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);
}
