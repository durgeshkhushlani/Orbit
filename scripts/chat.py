"""Multi-turn chat with Orbit's Retrieval Agent, backed by a LangGraph
checkpointer so conversation state persists across turns.

Usage: python scripts/chat.py
Type 'exit' or 'quit' to end the session.
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from orbit.graph.build import build_graph  # noqa: E402
from orbit.graph.checkpointer import get_checkpointer  # noqa: E402


def main() -> None:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)
        print(f"Orbit chat (thread {thread_id}). Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            result = graph.invoke({"messages": [HumanMessage(user_input)]}, config=config)
            print(f"\nOrbit: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
