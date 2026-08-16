import json

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from orbit.config import settings
from orbit.graph.build import build_graph
from orbit.retrieval.retriever import RetrievedChunk


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(settings, "orbit_allowed_dirs", str(root))
    return root


def _stub_generate_sequence(monkeypatch, responses):
    responses_iter = iter(responses)
    monkeypatch.setattr("orbit.graph.nodes.generate", lambda prompt: next(responses_iter))


def test_document_request_generates_file_grounded_in_retrieval(monkeypatch, allowed_root):
    destination = allowed_root / "summary.md"
    monkeypatch.setattr(
        "orbit.graph.nodes.retrieve",
        lambda query, n_results=5: [RetrievedChunk(text="cohesion info", source="doc.pdf", distance=0.5)],
    )
    _stub_generate_sequence(
        monkeypatch,
        [
            json.dumps({"format": "md", "destination": str(destination)}),
            "Generated summary content.",
        ],
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "doc-generate"}}

    result = graph.invoke(
        {"messages": [HumanMessage("save as markdown a summary of cohesion")], "sources": []},
        config=config,
    )

    assert "__interrupt__" not in result
    assert destination.exists()
    assert destination.read_text(encoding="utf-8") == "Generated summary content."
    assert result["sources"] == ["doc.pdf"]


def test_out_of_scope_destination_is_refused_without_writing(monkeypatch, tmp_path, allowed_root):
    outside_destination = tmp_path / "outside" / "summary.md"
    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: [])
    _stub_generate_sequence(
        monkeypatch, [json.dumps({"format": "md", "destination": str(outside_destination)})]
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "doc-out-of-scope"}}

    result = graph.invoke(
        {"messages": [HumanMessage("save as markdown a summary")], "sources": []}, config=config
    )

    assert "outside the folders" in result["messages"][-1].content
    assert not outside_destination.exists()


def test_unparseable_plan_asks_for_format_and_destination(monkeypatch, allowed_root):
    _stub_generate_sequence(monkeypatch, ["not valid json"])

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "doc-bad-plan"}}

    result = graph.invoke(
        {"messages": [HumanMessage("save as markdown something")], "sources": []}, config=config
    )

    assert "format" in result["messages"][-1].content.lower()


def test_document_request_has_no_confirm_gate(monkeypatch, allowed_root):
    """Document generation is ungated per the plan -- it should never pause,
    unlike File Agent's Confirm? gate."""
    destination = allowed_root / "report.docx"
    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: [])
    _stub_generate_sequence(
        monkeypatch,
        [json.dumps({"format": "docx", "destination": str(destination)}), "Report body."],
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "doc-no-confirm"}}

    result = graph.invoke(
        {"messages": [HumanMessage("generate a report and save as docx")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert destination.exists()
