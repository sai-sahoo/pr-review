"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { API_URL } from "@/lib/types";

// Home page: paste a PR URL, start a review, jump to its live page.
export default function HomePage() {
  const router = useRouter();
  const [prUrl, setPrUrl] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function startReview(e: React.SubmitEvent<HTMLFormElement>) {
    e.preventDefault(); // stop the browser's own form submit (a full page reload)
    setBusy(true);
    setError(null);
    try {
      const resp = await fetch(`${API_URL}/reviews`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pr_url: prUrl }),
      });
      const body = await resp.json();
      if (!resp.ok) {
        // Our 422 has detail as a string; Pydantic's 422 has a list of errors.
        setError(typeof body.detail === "string" ? body.detail : "Please enter a PR URL.");
        return;
      }
      router.push(`/reviews/${body.id}`); // 202 new, or 200 an existing review of this commit: go watch it
    } catch {
      setError("Could not reach the API. Is the backend running?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h1>pr-review</h1>
      <p className="muted">Paste a GitHub pull request URL and watch the agents review it live.</p>
      <form onSubmit={startReview} className="row">
        <input
          value={prUrl}
          onChange={(e) => setPrUrl(e.target.value)}
          placeholder="https://github.com/owner/repo/pull/123"
          required
        />
        <button disabled={busy}>{busy ? "Starting…" : "Review"}</button>
      </form>
      {error && <p className="error">{error}</p>}
    </>
  );
}
