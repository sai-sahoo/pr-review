"""Wire nodes into a graph:

    START -> fetch_pr -> triage -> specialist x N (parallel) -> aggregate -> verify -> skip_seen -> approve -> publish -> resolve_fixed -> END
                                \\-------- (empty plan) --------/                         \\-- (nothing to ask) --/

skip_seen drops what's already on the PR or was dismissed before.
approve pauses the run (interrupt) until a human decides what gets posted.
resolve_fixed marks the bot's threads resolved where the code they point at is gone.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    aggregate,
    approve,
    fetch_pr,
    publish,
    resolve_fixed,
    route_to_approval,
    route_to_specialists,
    skip_seen,
    specialist,
    triage,
    verify,
)
from app.graph.state import ReviewState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(ReviewState)  # the state schema every node shares

    builder.add_node("fetch_pr", fetch_pr)
    builder.add_node("triage", triage)
    builder.add_node("specialist", specialist)
    builder.add_node("aggregate", aggregate)
    builder.add_node("verify", verify)
    builder.add_node("skip_seen", skip_seen)
    builder.add_node("approve", approve)
    builder.add_node("publish", publish)
    builder.add_node("resolve_fixed", resolve_fixed)

    builder.add_edge(START, "fetch_pr")
    builder.add_edge("fetch_pr", "triage")
    # The router decides at runtime where to go. The list names every possible
    # destination so compile() can validate it and the diagram can draw it.
    builder.add_conditional_edges("triage", route_to_specialists, ["specialist", "aggregate"])
    # aggregate runs once, after every parallel specialist in the step has finished.
    builder.add_edge("specialist", "aggregate")
    builder.add_edge("aggregate", "verify")
    builder.add_edge("verify", "skip_seen")
    builder.add_conditional_edges("skip_seen", route_to_approval, ["approve", "publish"])
    builder.add_edge("approve", "publish")
    builder.add_edge("publish", "resolve_fixed")
    builder.add_edge("resolve_fixed", END)

    # compile() validates the wiring (no dangling nodes, no missing edges)
    # and returns a runnable with .invoke() / .stream().
    # With a checkpointer, every step's state is saved under the run's thread_id.
    # approve's interrupt() needs one: a paused run *is* its last checkpoint.
    return builder.compile(checkpointer=checkpointer)
