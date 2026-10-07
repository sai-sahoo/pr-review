"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { FindingCard } from "@/components/finding-card";
import { API_URL } from "@/lib/types";
import { useReviewEvents } from "@/lib/use-review-events";

// /reviews/<id>: the folder name [id] makes that part of the URL a parameter.
export default function ReviewPage() {
  const { id } = useParams<{ id: string }>();
  const review = useReviewEvents(id);
  const { pr, plan, agents, checks } = review;

  // Kept in the server's order, not re-sorted: the approval sends positions
  // in this list (the record's `findings`). It's already sorted by severity.
  const kept = checks?.filter((c) => c.kept) ?? [];
  const dropped = checks?.filter((c) => !c.kept) ?? [];

  // Approval: every finding starts ticked; you untick the ones to dismiss.
  const waiting = review.status === "waiting";
  const [dismissed, setDismissed] = useState<Set<number>>(new Set());
  const [sending, setSending] = useState(false);
  const [approvalError, setApprovalError] = useState<string>();

  function toggle(i: number) {
    const next = new Set(dismissed); // a new Set, never the old one changed: React compares by identity
    if (next.has(i)) next.delete(i);
    else next.add(i);
    setDismissed(next);
  }

  async function sendApproval() {
    setSending(true);
    setApprovalError(undefined);
    const approved = kept.map((_, i) => i).filter((i) => !dismissed.has(i));
    try {
      const resp = await fetch(`${API_URL}/reviews/${id}/approval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved }),
      });
      // No state change on success: the event stream, still open, brings
      // "running", then approve, publish and done, as the worker carries on.
      if (!resp.ok) setApprovalError((await resp.json()).detail ?? `HTTP ${resp.status}`);
    } catch {
      setApprovalError("Could not reach the API.");
    } finally {
      setSending(false);
    }
  }

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
      {waiting && (
        <section className="card">
          <p>
            <strong>Waiting for your approval.</strong> Untick anything that shouldn&apos;t go on the PR.
          </p>
          <p className="row">
            <button onClick={sendApproval} disabled={sending}>
              {kept.length - dismissed.size === 0
                ? "Dismiss all, post nothing"
                : `Post ${kept.length - dismissed.size} of ${kept.length} to GitHub`}
            </button>
            {approvalError && <span className="error">{approvalError}</span>}
          </p>
        </section>
      )}
      {review.approved !== undefined && (
        <p className="muted">
          Approved {review.approved} of {kept.length} findings for posting.
        </p>
      )}
      {review.published && (
        <p className="muted">
          {review.published.url ? (
            <a href={review.published.url} target="_blank" rel="noreferrer">
              View the review on GitHub →
            </a>
          ) : (
            review.published.note
          )}
        </p>
      )}

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
          {kept.map((c, i) => (
            <FindingCard
              key={i}
              finding={c.finding}
              {...(waiting && { selected: !dismissed.has(i), onToggle: () => toggle(i) })}
            />
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
