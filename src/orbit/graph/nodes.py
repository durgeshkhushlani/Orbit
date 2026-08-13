from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from orbit.generation.prompt import build_prompt
from orbit.graph.state import OrbitState
from orbit.llm.ollama_client import generate
from orbit.retrieval.retriever import retrieve


def _latest_query(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("No human message found in state")


def supervisor_node(state: OrbitState) -> dict:
    """Routes to a specialist agent. Retrieval is the only specialist so far;
    this is the seam where routing logic grows as more agents are added."""
    return {}


def retrieval_node(state: OrbitState) -> dict:
    """Retrieve relevant chunks for the latest query, ground a prompt in them,
    and generate an answer."""
    query = _latest_query(state["messages"])
    chunks = retrieve(query)

    prompt = build_prompt(query, chunks)
    answer = generate(prompt)
    sources = list(dict.fromkeys(chunk.source for chunk in chunks))

    return {"messages": [AIMessage(content=answer)], "sources": sources}
