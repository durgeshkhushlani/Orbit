import json

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orbit.config import settings
from orbit.graph.build import build_graph
from orbit.web_agent.search import SearchResult


@pytest.fixture
def allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(settings, "orbit_allowed_dirs", str(root))
    return root


def _stub_search(monkeypatch, results):
    monkeypatch.setattr("orbit.graph.nodes.search_web", lambda query: results)


def _stub_extract(monkeypatch, text="Extracted page text."):
    monkeypatch.setattr("orbit.graph.nodes.extract_content", lambda url: text)


SAMPLE_RESULTS = [SearchResult(title="Result", url="https://example.com/a", snippet="A snippet.")]


def test_search_only_request_answers_without_confirm(monkeypatch, allowed_root):
    monkeypatch.setattr(
        "orbit.graph.nodes.generate",
        lambda prompt: json.dumps({"query": "python 3.14 features", "save_as": None, "destination": None})
        if "Extract a web search request" in prompt
        else "Python 3.14 adds ...",
    )
    _stub_search(monkeypatch, SAMPLE_RESULTS)
    _stub_extract(monkeypatch)

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "web-search"}}

    result = graph.invoke(
        {"messages": [HumanMessage("search the web for python 3.14 features")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert result["messages"][-1].content == "Python 3.14 adds ..."
    assert result["sources"] == ["https://example.com/a"]


def test_save_request_pauses_for_confirm_then_saves_and_indexes(monkeypatch, allowed_root):
    destination = allowed_root / "research.md"
    monkeypatch.setattr(
        "orbit.graph.nodes.generate",
        lambda prompt: (
            json.dumps({"query": "python 3.14 features", "save_as": "md", "destination": str(destination)})
            if "Extract a web search request" in prompt
            else "# Python 3.14\n\nSome findings."
        ),
    )
    _stub_search(monkeypatch, SAMPLE_RESULTS)
    _stub_extract(monkeypatch)
    indexed = {}

    def _stub_index_file(path):
        indexed["path"] = path
        return {"files_loaded": 1, "chunks_indexed": 1}

    monkeypatch.setattr("orbit.graph.nodes.index_file", _stub_index_file)

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "web-save"}}

    paused = graph.invoke(
        {
            "messages": [HumanMessage("search the web for python 3.14 features and save as markdown")],
            "sources": [],
        },
        config=config,
    )

    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["type"] == "confirm"
    assert not destination.exists()

    resumed = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in resumed
    assert destination.exists()
    assert destination.read_text() == "# Python 3.14\n\nSome findings."
    assert indexed["path"] == destination.resolve()
    assert "Saved" in resumed["messages"][-1].content


def test_declining_save_confirm_does_not_write_or_index(monkeypatch, allowed_root):
    destination = allowed_root / "research.md"
    monkeypatch.setattr(
        "orbit.graph.nodes.generate",
        lambda prompt: json.dumps(
            {"query": "python 3.14 features", "save_as": "md", "destination": str(destination)}
        ),
    )
    _stub_search(monkeypatch, SAMPLE_RESULTS)
    _stub_extract(monkeypatch)
    monkeypatch.setattr("orbit.graph.nodes.index_file", lambda path: pytest.fail("should not index"))

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "web-save-decline"}}

    graph.invoke(
        {
            "messages": [HumanMessage("search the web for python 3.14 features and save as markdown")],
            "sources": [],
        },
        config=config,
    )
    resumed = graph.invoke(Command(resume="no"), config=config)

    assert resumed["messages"][-1].content == "Okay, I won't do that."
    assert not destination.exists()


def test_out_of_scope_destination_is_refused_without_ever_prompting_confirm(monkeypatch, tmp_path, allowed_root):
    outside_destination = tmp_path / "research.md"
    monkeypatch.setattr(
        "orbit.graph.nodes.generate",
        lambda prompt: json.dumps(
            {"query": "python 3.14 features", "save_as": "md", "destination": str(outside_destination)}
        ),
    )
    _stub_search(monkeypatch, SAMPLE_RESULTS)
    _stub_extract(monkeypatch)

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "web-out-of-scope"}}

    result = graph.invoke(
        {
            "messages": [HumanMessage("search the web for python 3.14 features and save as markdown")],
            "sources": [],
        },
        config=config,
    )

    assert "__interrupt__" not in result
    assert "outside the folders" in result["messages"][-1].content


def test_no_results_reports_gracefully_without_confirm(monkeypatch, allowed_root):
    monkeypatch.setattr(
        "orbit.graph.nodes.generate",
        lambda prompt: json.dumps({"query": "an obscure query", "save_as": None, "destination": None}),
    )
    _stub_search(monkeypatch, [])

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "web-no-results"}}

    result = graph.invoke(
        {"messages": [HumanMessage("search the web for an obscure query")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert "couldn't find anything" in result["messages"][-1].content


def test_unparseable_plan_asks_to_rephrase_without_prompting_confirm(monkeypatch, allowed_root):
    monkeypatch.setattr("orbit.graph.nodes.generate", lambda prompt: "not valid json")

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "web-bad-plan"}}

    result = graph.invoke(
        {"messages": [HumanMessage("search the web for something")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert "rephrase" in result["messages"][-1].content.lower()
