// The shapes the backend sends us. They mirror the Pydantic models in
// backend/app/schemas.py and the event dicts in backend/app/api/events.py.
// TypeScript types vanish at runtime: they only help the editor catch typos.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Severity = "critical" | "high" | "medium" | "low";

export const SEVERITY_RANK: Record<Severity, number> = { critical: 0, high: 1, medium: 2, low: 3 };

export type Finding = {
  file: string;
  line: number;
  severity: Severity;
  category: string;
  title: string;
  explanation: string;
  suggestion: string;
};

export type FindingCheck = {
  finding: Finding;
  kept: boolean;
  stage: "grounding" | "verifier";
  confidence: number | null;
  reason: string;
};

// One event from GET /reviews/{id}/events. `type` is also the SSE "event:" name.
// A union with a shared `type` field lets TypeScript know which other fields
// exist once you've checked `type` (and `node`), like a match statement.
export type ReviewEvent =
  | { type: "status"; status: "running" }
  | { type: "node"; node: "fetch_pr"; number: number; title: string; files: string[]; skipped: string[] }
  | { type: "node"; node: "triage"; specialists: string[]; reason: string }
  | { type: "node"; node: "specialist"; findings: Finding[]; log: string[] }
  | { type: "node"; node: "aggregate"; count: number }
  | { type: "node"; node: "verify"; kept: number; checks: FindingCheck[] }
  | { type: "agent"; agent: string; tool: string; args: Record<string, unknown> }
  | { type: "agent"; agent: string; submitted: number }
  | { type: "done"; kept: number }
  | { type: "failed"; error: string };

export const EVENT_TYPES = ["status", "node", "agent", "done", "failed"] as const;
