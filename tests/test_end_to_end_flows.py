"""Multi-turn, multi-agent conversations exercised through the full compiled
graph on a single thread -- distinct from the per-agent test files, which
each test one specialist in isolation. These prove the Supervisor keeps
routing correctly turn to turn, conversation history accumulates via the
checkpointer, and a completed Confirm?/Clarify? cycle doesn't leave the
thread stuck. All external calls (LLM, retrieval, web, SMTP, indexing) are
stubbed so the suite stays offline for CI."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orbit.config import settings
from orbit.graph.build import build_graph
from orbit.retrieval.retriever import RetrievedChunk
from orbit.web_agent.search import SearchResult


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(settings, "orbit_allowed_dirs", str(root))
    return root


def test_research_then_report_flow(monkeypatch, allowed_root):
    """Turn 1: a grounded retrieval answer. Turn 2, same thread: generate a
    document from that same context. Confirms conversation history
    accumulates across turns and each turn routes to the right specialist."""
    destination = allowed_root / "cohesion_summary.md"
    chunks = [RetrievedChunk(text="Cohesion measures how related a module's responsibilities are.", source="notes.pdf", distance=0.4)]
    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: chunks)

    responses = iter(
        [
            "Cohesion is how focused a module's responsibilities are.",
            json.dumps({"format": "md", "destination": str(destination)}),
            "# Cohesion\n\nCohesion is how focused a module's responsibilities are.",
        ]
    )
    monkeypatch.setattr("orbit.graph.nodes.generate", lambda prompt: next(responses))

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "research-then-report"}}

    turn1 = graph.invoke({"messages": [HumanMessage("what is cohesion")], "sources": []}, config=config)
    assert "__interrupt__" not in turn1
    assert turn1["messages"][-1].content == "Cohesion is how focused a module's responsibilities are."

    turn2 = graph.invoke(
        {"messages": [HumanMessage("save as markdown a summary of that")], "sources": []}, config=config
    )

    assert "__interrupt__" not in turn2
    assert destination.exists()
    assert destination.read_text(encoding="utf-8") == "# Cohesion\n\nCohesion is how focused a module's responsibilities are."

    human_turns = [m for m in turn2["messages"] if isinstance(m, HumanMessage)]
    ai_turns = [m for m in turn2["messages"] if isinstance(m, AIMessage)]
    assert len(human_turns) == 2
    assert len(ai_turns) == 2


def test_rename_then_retrieve_flow(monkeypatch, allowed_root):
    """Turn 1: rename a file through the Confirm? pause/resume cycle. Turn 2,
    same thread: an unrelated retrieval question. Confirms the graph is fully
    usable again after a completed interrupt cycle, not stuck on File Agent."""
    source = allowed_root / "draft.pdf"
    source.write_text("content")

    plan_response = json.dumps({"action": "rename", "source": str(source), "destination": "final.pdf"})
    monkeypatch.setattr("orbit.graph.nodes.generate", lambda prompt: plan_response)

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "rename-then-retrieve"}}

    paused = graph.invoke(
        {"messages": [HumanMessage("rename draft.pdf to final.pdf")], "sources": []}, config=config
    )
    assert "__interrupt__" in paused

    resumed = graph.invoke(Command(resume="yes"), config=config)
    assert "__interrupt__" not in resumed
    assert (allowed_root / "final.pdf").exists()

    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: [])
    monkeypatch.setattr("orbit.graph.nodes.generate", lambda prompt: "I don't know.")

    turn2 = graph.invoke({"messages": [HumanMessage("what is the meaning of life")], "sources": []}, config=config)

    assert "__interrupt__" in turn2
    assert turn2["__interrupt__"][0].value["type"] == "clarify"


def test_web_research_then_email_flow(monkeypatch, allowed_root):
    """Turn 1: search the web and save results, through Confirm?, then
    auto-index. Turn 2, same thread: email that same saved file as an
    attachment, through its own Confirm? -- proves a file the Web Agent just
    created is immediately usable (in-scope) by the Email Agent.

    generate() is stubbed by inspecting the prompt rather than a plain call
    sequence: a node resumed via Command(resume=...) re-executes from the top
    (interrupt() only skips its own re-pause), so any generate() call before
    the interrupt -- like plan extraction -- fires again on resume.
    """
    destination = allowed_root / "research.md"
    monkeypatch.setattr(
        "orbit.graph.nodes.search_web",
        lambda query: [SearchResult(title="R", url="https://example.com", snippet="A snippet.")],
    )
    monkeypatch.setattr("orbit.graph.nodes.extract_content", lambda url: "Extracted findings.")
    monkeypatch.setattr("orbit.graph.nodes.index_file", lambda path: {"files_loaded": 1, "chunks_indexed": 1})

    def web_generate(prompt):
        if "Extract a web search request" in prompt:
            return json.dumps({"query": "orbit project", "save_as": "md", "destination": str(destination)})
        return "# Orbit findings\n\nExtracted findings."

    monkeypatch.setattr("orbit.graph.nodes.generate", web_generate)

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "web-then-email"}}

    paused = graph.invoke(
        {"messages": [HumanMessage("search the web for orbit project and save as markdown")], "sources": []},
        config=config,
    )
    assert "__interrupt__" in paused

    resumed = graph.invoke(Command(resume="yes"), config=config)
    assert "__interrupt__" not in resumed
    assert destination.exists()

    monkeypatch.setattr(
        "orbit.graph.nodes.generate",
        lambda prompt: json.dumps(
            {"to": "friend@example.com", "subject": "Findings", "body": "See attached.", "attachment": str(destination)}
        ),
    )
    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: [])
    sent = {}
    monkeypatch.setattr(
        "orbit.graph.nodes.send_email",
        lambda to, subject, body, attachment=None: sent.update(to=to, attachment=attachment),
    )

    paused_email = graph.invoke(
        {"messages": [HumanMessage("email research.md to friend@example.com")], "sources": []}, config=config
    )
    assert "__interrupt__" in paused_email

    resumed_email = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in resumed_email
    assert sent["to"] == "friend@example.com"
    assert sent["attachment"] == destination.resolve()
