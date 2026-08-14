import json

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orbit.config import settings
from orbit.graph.build import build_graph


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(settings, "orbit_allowed_dirs", str(root))
    return root


def _stub_generate(monkeypatch, response):
    monkeypatch.setattr("orbit.graph.nodes.generate", lambda prompt: response)


def test_rename_request_routes_to_file_agent_and_pauses_for_confirm(monkeypatch, allowed_root):
    source = allowed_root / "invoice_draft.pdf"
    source.write_text("content")
    _stub_generate(
        monkeypatch,
        json.dumps({"action": "rename", "source": str(source), "destination": "invoice_final.pdf"}),
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "rename-confirm"}}

    paused = graph.invoke(
        {"messages": [HumanMessage("rename invoice_draft.pdf to invoice_final.pdf")], "sources": []},
        config=config,
    )

    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["type"] == "confirm"
    assert source.exists()

    resumed = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in resumed
    assert (allowed_root / "invoice_final.pdf").exists()
    assert not source.exists()


def test_declining_confirm_leaves_file_untouched(monkeypatch, allowed_root):
    source = allowed_root / "invoice_draft.pdf"
    source.write_text("content")
    _stub_generate(
        monkeypatch,
        json.dumps({"action": "rename", "source": str(source), "destination": "invoice_final.pdf"}),
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "rename-decline"}}

    graph.invoke(
        {"messages": [HumanMessage("rename invoice_draft.pdf to invoice_final.pdf")], "sources": []},
        config=config,
    )
    resumed = graph.invoke(Command(resume="no"), config=config)

    assert resumed["messages"][-1].content == "Okay, I won't do that."
    assert source.exists()
    assert not (allowed_root / "invoice_final.pdf").exists()


def test_out_of_scope_source_is_refused_without_ever_prompting_confirm(monkeypatch, tmp_path, allowed_root):
    outside_source = tmp_path / "outside.pdf"
    outside_source.write_text("content")
    _stub_generate(
        monkeypatch,
        json.dumps({"action": "rename", "source": str(outside_source), "destination": "renamed.pdf"}),
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "out-of-scope"}}

    result = graph.invoke(
        {"messages": [HumanMessage("rename outside.pdf to renamed.pdf")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert "outside the folders" in result["messages"][-1].content
    assert outside_source.exists()


def test_unparseable_plan_asks_to_rephrase_without_prompting_confirm(monkeypatch, allowed_root):
    _stub_generate(monkeypatch, "not valid json")

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "bad-plan"}}

    result = graph.invoke(
        {"messages": [HumanMessage("rename something")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert "rephrase" in result["messages"][-1].content.lower()


def test_non_file_query_still_routes_to_retrieval_agent(monkeypatch, allowed_root):
    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: [])
    _stub_generate(monkeypatch, "retrieval answer")

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "retrieval-route"}}

    result = graph.invoke(
        {"messages": [HumanMessage("what is cohesion")], "sources": []}, config=config
    )

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["type"] == "clarify"
