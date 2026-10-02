"""Wire nodes into a graph:  START -> fetch_pr -> review -> END"""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import fetch_pr, review
from app.graph.state import ReviewState


def build_graph():
    builder = StateGraph(ReviewState)  # the state schema every node shares

    builder.add_node("fetch_pr", fetch_pr)
    builder.add_node("review", review)

    builder.add_edge(START, "fetch_pr")
    builder.add_edge("fetch_pr", "review")
    builder.add_edge("review", END)

    # compile() validates the wiring (no dangling nodes, no missing edges)
    # and returns a runnable with .invoke() / .stream()
    return builder.compile()
