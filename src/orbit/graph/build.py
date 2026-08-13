from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orbit.graph.nodes import retrieval_node, supervisor_node
from orbit.graph.state import OrbitState


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Wire Supervisor -> Retrieval Agent and compile with the given checkpointer.

    Only one specialist exists so far, so Supervisor routes to it
    unconditionally; this is the seam where more agents (File, Document,
    Email, Web) get added in later days.
    """
    graph = StateGraph(OrbitState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval_agent", retrieval_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "retrieval_agent")
    graph.add_edge("retrieval_agent", END)

    return graph.compile(checkpointer=checkpointer)
