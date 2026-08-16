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


def _stub_retrieve(monkeypatch, chunks=None):
    monkeypatch.setattr("orbit.graph.nodes.retrieve", lambda query, n_results=5: chunks or [])


def test_email_request_pauses_for_confirm_then_sends(monkeypatch, allowed_root):
    attachment = allowed_root / "resume.pdf"
    attachment.write_text("resume content")
    _stub_retrieve(monkeypatch)
    _stub_generate(
        monkeypatch,
        json.dumps(
            {
                "to": "me@example.com",
                "subject": "My resume",
                "body": "Attached.",
                "attachment": str(attachment),
            }
        ),
    )
    sent = {}
    monkeypatch.setattr(
        "orbit.graph.nodes.send_email",
        lambda to, subject, body, attachment=None: sent.update(
            to=to, subject=subject, body=body, attachment=attachment
        ),
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "email-confirm"}}

    paused = graph.invoke(
        {"messages": [HumanMessage("email my resume to myself")], "sources": []}, config=config
    )

    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value["type"] == "confirm"
    assert not sent

    resumed = graph.invoke(Command(resume="yes"), config=config)

    assert "__interrupt__" not in resumed
    assert sent["to"] == "me@example.com"
    assert sent["attachment"] == attachment.resolve()
    assert "Sent" in resumed["messages"][-1].content


def test_declining_confirm_does_not_send(monkeypatch, allowed_root):
    _stub_retrieve(monkeypatch)
    _stub_generate(
        monkeypatch,
        json.dumps({"to": "friend@example.com", "subject": "Hi", "body": "Hello!", "attachment": None}),
    )
    sent = {}
    monkeypatch.setattr(
        "orbit.graph.nodes.send_email",
        lambda to, subject, body, attachment=None: sent.update(to=to),
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "email-decline"}}

    graph.invoke({"messages": [HumanMessage("email friend@example.com to say hi")], "sources": []}, config=config)
    resumed = graph.invoke(Command(resume="no"), config=config)

    assert resumed["messages"][-1].content == "Okay, I won't send that."
    assert not sent


def test_out_of_scope_attachment_is_refused_without_ever_prompting_confirm(monkeypatch, tmp_path, allowed_root):
    outside_attachment = tmp_path / "secret.pdf"
    outside_attachment.write_text("content")
    _stub_retrieve(monkeypatch)
    _stub_generate(
        monkeypatch,
        json.dumps(
            {
                "to": "friend@example.com",
                "subject": "Doc",
                "body": "See attached.",
                "attachment": str(outside_attachment),
            }
        ),
    )

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "email-out-of-scope"}}

    result = graph.invoke(
        {"messages": [HumanMessage("email secret.pdf to friend@example.com")], "sources": []}, config=config
    )

    assert "__interrupt__" not in result
    assert "outside the folders" in result["messages"][-1].content


def test_unparseable_plan_asks_to_rephrase_without_prompting_confirm(monkeypatch, allowed_root):
    _stub_retrieve(monkeypatch)
    _stub_generate(monkeypatch, "not valid json")

    graph = build_graph(InMemorySaver())
    config = {"configurable": {"thread_id": "email-bad-plan"}}

    result = graph.invoke({"messages": [HumanMessage("email something")], "sources": []}, config=config)

    assert "__interrupt__" not in result
    assert "specify" in result["messages"][-1].content.lower()
