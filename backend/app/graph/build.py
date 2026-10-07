"""Wire nodes into a graph:

    START -> fetch_pr -> triage -> specialist x N (parallel) -> aggregate -> verify -> approve -> publish -> END
                                \\-------- (empty plan) --------/            \\-- (nothing to ask) --/

approve pauses the run (interrupt) until a human decides what gets posted.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    aggregate,
    approve,
    fetch_pr,
    publish,
    route_after_verify,
    route_to_specialists,
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
    builder.add_node("approve", approve)
    builder.add_node("publish", publish)

    builder.add_edge(START, "fetch_pr")
    builder.add_edge("fetch_pr", "triage")
    # The router decides at runtime where to go. The list names every possible
    # destination so compile() can validate it and the diagram can draw it.
    builder.add_conditional_edges("triage", route_to_specialists, ["specialist", "aggregate"])
    # aggregate runs once, after every parallel specialist in the step has finished.
    builder.add_edge("specialist", "aggregate")
    builder.add_edge("aggregate", "verify")
    builder.add_conditional_edges("verify", route_after_verify, ["approve", "publish"])
    builder.add_edge("approve", "publish")
    builder.add_edge("publish", END)

    # compile() validates the wiring (no dangling nodes, no missing edges)
    # and returns a runnable with .invoke() / .stream().
    # With a checkpointer, every step's state is saved under the run's thread_id.
    # approve's interrupt() needs one: a paused run *is* its last checkpoint.
    return builder.compile(checkpointer=checkpointer)
