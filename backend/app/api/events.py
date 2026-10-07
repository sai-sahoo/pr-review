"""Turn graph stream chunks into small JSON events for the browser.

A node's raw update holds Pydantic objects (a whole PullRequest with every
hunk). Clients need a summary of what just happened, as plain JSON.
"""

from typing import Any

TERMINAL = {"done", "failed"}  # after one of these, the stream ends


def node_event(node: str, update: dict) -> dict[str, Any]:
    """One `updates` chunk entry: a node has finished."""
    event: dict[str, Any] = {"type": "node", "node": node}
    if node == "fetch_pr":
        pr = update["pr"]
        event |= {
            "number": pr.number,
            "title": pr.title,
            "files": [f.path for f in pr.files],
            "skipped": [s.path for s in pr.skipped],
        }
    elif node == "triage":
        plan = update["plan"]
        event |= {"specialists": plan.specialists, "reason": plan.reason}
    elif node == "specialist":
        # mode="json" turns enums, datetimes etc. into JSON-safe values
        event |= {
            "findings": [f.model_dump(mode="json") for f in update["raw_findings"]],
            "log": update["tool_log"],
        }
    elif node == "aggregate":
        event |= {"count": len(update["findings"])}
    elif node == "verify":
        event |= {
            "kept": len(update["verified"]),
            "checks": [c.model_dump(mode="json") for c in update["checks"]],
        }
    elif node == "approve":
        event |= {"approved": len(update["approved"])}
    elif node == "publish":
        event |= {"review_url": update["review_url"], "note": update["publish_note"]}
    return event


def agent_event(chunk: dict) -> dict[str, Any]:
    """One `custom` chunk, sent by agent.py's emit while a specialist works:
    {"agent", "tool", "args"} for a tool call, {"agent", "submitted"} at the end.
    """
    return {"type": "agent", **chunk}
