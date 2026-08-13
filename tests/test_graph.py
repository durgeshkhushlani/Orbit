from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orbit.graph.build import build_graph
from orbit.retrieval.retriever import RetrievedChunk


def _stub_retrieve_and_generate(monkeypatch, chunk_batches, answer="stub answer"):
    batches = iter(chunk_batches)
    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: next(batches))
    monkeypatch.setattr("orbit.graph.nodes.generate", lambda prompt: answer)


def test_high_confidence_query_answers_without_interrupt(monkeypatch):
    chunks = [RetrievedChunk(text="cohesion info", source="doc.pdf", distance=0.5)]
    _stub_retrieve_and_generate(monkeypatch, [chunks])

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "high-confidence"}}

    result = graph.invoke(
        {"messages": [HumanMessage("what is cohesion")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "stub answer"
    assert result["sources"] == ["doc.pdf"]


def test_low_confidence_query_pauses_and_resumes(monkeypatch):
    low_confidence_chunks = [RetrievedChunk(text="unrelated", source="a.pdf", distance=1.8)]
    resumed_chunks = [RetrievedChunk(text="cocomo info", source="b.pdf", distance=0.5)]
    _stub_retrieve_and_generate(monkeypatch, [low_confidence_chunks, resumed_chunks])

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "low-confidence"}}

    paused = graph.invoke(
        {"messages": [HumanMessage("cake recipe")], "sources": []}, config=config
    )

    assert "__interrupt__" in paused
    assert "confident" in paused["__interrupt__"][0].value["question"].lower()

    resumed = graph.invoke(Command(resume="cocomo cost estimation"), config=config)

    assert "__interrupt__" not in resumed
    assert resumed["messages"][-1].content == "stub answer"
    assert resumed["sources"] == ["b.pdf"]


def test_resume_does_not_re_invoke_supervisor(monkeypatch):
    """Resuming a Clarify interrupt must continue inside the paused node, not
    restart the graph from Supervisor -- that's the whole point of using
    interrupt()/Command(resume=...) instead of re-routing from scratch."""
    low_confidence_chunks = [RetrievedChunk(text="unrelated", source="a.pdf", distance=1.8)]
    resumed_chunks = [RetrievedChunk(text="cocomo info", source="b.pdf", distance=0.5)]
    _stub_retrieve_and_generate(monkeypatch, [low_confidence_chunks, resumed_chunks])

    call_count = {"supervisor": 0}

    def counting_supervisor(state):
        call_count["supervisor"] += 1
        return {}

    monkeypatch.setattr("orbit.graph.build.supervisor_node", counting_supervisor)

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "no-restart"}}

    graph.invoke({"messages": [HumanMessage("cake recipe")], "sources": []}, config=config)
    graph.invoke(Command(resume="cocomo cost estimation"), config=config)

    assert call_count["supervisor"] == 1
