// Everything the review page shows, rebuilt from the event stream.
//
// reviewReducer(state, event) -> new state is a pure function: no fetching,
// no timers, it never changes the old state. The page's whole state is then
// just "all events so far, folded together", like Python's
// functools.reduce(review_reducer, events, initial_state). That's also why a
// page reload works: the server replays every event from the start.

import type { Finding, FindingCheck, ReviewEvent } from "./types";

export type AgentState = {
  status: "working" | "done";
  tools: string[]; // one line per tool call, e.g. "read_file app/db.py lines 1-40"
  submitted?: number; // how many findings it reported
};

export type ReviewState = {
  status: "connecting" | "running" | "waiting" | "done" | "failed";
  pr?: { number: number; title: string; files: string[]; skipped: string[] };
  plan?: { specialists: string[]; reason: string };
  agents: Record<string, AgentState>;
  rawFindings: Finding[]; // what the specialists reported, before the verifier
  checks?: FindingCheck[]; // the verifier's decision on each finding
  approved?: number; // how many findings a human let through at the approval step
  published?: { url: string | null; note: string }; // posted to the PR, or why not
  error?: string;
};

export const initialState: ReviewState = { status: "connecting", agents: {}, rawFindings: [] };

// An action is an event from the server, or "reset" from our own code.
export type Action = ReviewEvent | { type: "reset" };

export function reviewReducer(state: ReviewState, action: Action): ReviewState {
  switch (action.type) {
    case "reset":
      return initialState;
    case "status":
      return { ...state, status: action.status };
    case "node":
      return applyNode(state, action);
    case "agent": {
      const agent = state.agents[action.agent] ?? { status: "working", tools: [] };
      const updated: AgentState =
        "submitted" in action
          ? { ...agent, status: "done", submitted: action.submitted }
          : { ...agent, tools: [...agent.tools, describeTool(action.tool, action.args)] };
      return { ...state, agents: { ...state.agents, [action.agent]: updated } };
    }
    case "done":
      return { ...state, status: "done" };
    case "failed":
      return { ...state, status: "failed", error: action.error };
  }
}

function applyNode(state: ReviewState, event: Extract<ReviewEvent, { type: "node" }>): ReviewState {
  switch (event.node) {
    case "fetch_pr":
      return { ...state, pr: { number: event.number, title: event.title, files: event.files, skipped: event.skipped } };
    case "triage":
      // The chosen specialists all start at once (the Send fan-out), so mark each one as working.
      return {
        ...state,
        plan: { specialists: event.specialists, reason: event.reason },
        agents: Object.fromEntries(event.specialists.map((name) => [name, { status: "working", tools: [] }])),
      };
    case "specialist":
      return { ...state, rawFindings: [...state.rawFindings, ...event.findings] };
    case "verify":
      return { ...state, checks: event.checks };
    case "approve":
      return { ...state, approved: event.approved };
    case "publish":
      return { ...state, published: { url: event.review_url, note: event.note } };
    default:
      return state; // aggregate: nothing new to show
  }
}

function describeTool(tool: string, args: Record<string, unknown>): string {
  if (tool === "read_file") return `read_file ${args.path} lines ${args.start_line}-${args.end_line}`;
  return `${tool} ${JSON.stringify(args)}`;
}
