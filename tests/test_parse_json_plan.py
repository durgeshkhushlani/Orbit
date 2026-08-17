import json

import pytest

from orbit.graph.nodes import _parse_json_plan


def test_parses_bare_json():
    assert _parse_json_plan('{"format": "md", "destination": "x.md"}') == {
        "format": "md",
        "destination": "x.md",
    }


def test_parses_json_wrapped_in_labeled_code_fence():
    """Real models sometimes wrap strict-JSON answers in ```json fences
    despite being told "ONLY the JSON object, nothing else" -- caught via a
    live end-to-end test against the real Ollama model."""
    response = '```json\n{"format": "md", "destination": "x.md"}\n```'
    assert _parse_json_plan(response) == {"format": "md", "destination": "x.md"}


def test_parses_json_wrapped_in_bare_code_fence():
    response = '```\n{"format": "md", "destination": "x.md"}\n```'
    assert _parse_json_plan(response) == {"format": "md", "destination": "x.md"}


def test_raises_on_response_with_no_json_object():
    with pytest.raises(json.JSONDecodeError):
        _parse_json_plan("sorry, I don't understand")
