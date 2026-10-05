"""Wire nodes into a graph:

    START -> fetch_pr -> triage -> specialist x N (parallel) -> aggregate -> verify -> END
                                \\-------- (empty plan) --------/
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import aggregate, fetch_pr, route_to_specialists, specialist, triage, verify
from app.graph.state import ReviewState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(ReviewState)  # the state schema every node shares

    builder.add_node("fetch_pr", fetch_pr)
    builder.add_node("triage", triage)
    builder.add_node("specialist", specialist)
    builder.add_node("aggregate", aggregate)
    builder.add_node("verify", verify)

    builder.add_edge(START, "fetch_pr")
    builder.add_edge("fetch_pr", "triage")
    # The router decides at runtime where to go. The list names every possible
    # destination so compile() can validate it and the diagram can draw it.
    builder.add_conditional_edges("triage", route_to_specialists, ["specialist", "aggregate"])
    # aggregate runs once, after every parallel specialist in the step has finished.
    builder.add_edge("specialist", "aggregate")
    builder.add_edge("aggregate", "verify")
    builder.add_edge("verify", END)

    # compile() validates the wiring (no dangling nodes, no missing edges)
    # and returns a runnable with .invoke() / .stream().
    # With a checkpointer, every step's state is saved under the run's thread_id.
    return builder.compile(checkpointer=checkpointer)
