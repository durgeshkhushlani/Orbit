from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orbit.graph.nodes import (
    document_agent_node,
    file_agent_node,
    retrieval_node,
    route_after_supervisor,
    supervisor_node,
)
from orbit.graph.state import OrbitState


def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Wire Supervisor -> {Retrieval Agent, File Agent, Document Agent} and
    compile with the given checkpointer. Supervisor routes conditionally
    between specialists; Email/Web agents get added to the same routing
    table in later days.
    """
    graph = StateGraph(OrbitState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval_agent", retrieval_node)
    graph.add_node("file_agent", file_agent_node)
    graph.add_node("document_agent", document_agent_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "retrieval_agent": "retrieval_agent",
            "file_agent": "file_agent",
            "document_agent": "document_agent",
        },
    )
    graph.add_edge("retrieval_agent", END)
    graph.add_edge("file_agent", END)
    graph.add_edge("document_agent", END)

    return graph.compile(checkpointer=checkpointer)
