from orbit.web_agent.extract import extract_content


def test_extract_content_returns_extracted_text(monkeypatch):
    monkeypatch.setattr("orbit.web_agent.extract.trafilatura.fetch_url", lambda url: "<html>raw</html>")
    monkeypatch.setattr(
        "orbit.web_agent.extract.trafilatura.extract",
        lambda downloaded, include_comments, include_tables: "Extracted body text.",
    )

    assert extract_content("https://example.com") == "Extracted body text."


def test_extract_content_returns_none_when_fetch_fails(monkeypatch):
    monkeypatch.setattr("orbit.web_agent.extract.trafilatura.fetch_url", lambda url: None)

    assert extract_content("https://example.com/missing") is None
