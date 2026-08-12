from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class OrbitState(TypedDict):
    """Shared state threaded through the Supervisor and specialist agent nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[str]
