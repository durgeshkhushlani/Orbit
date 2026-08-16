from orbit.web_agent.search import SearchResult, search_web


def test_search_web_maps_ddgs_results(monkeypatch):
    raw_results = [
        {"title": "Python", "href": "https://python.org", "body": "The Python language."},
        {"title": "Docs", "href": "https://docs.python.org", "body": "Python documentation."},
    ]

    class FakeDDGS:
        def text(self, query, max_results=5):
            assert query == "python"
            assert max_results == 5
            return raw_results

    monkeypatch.setattr("orbit.web_agent.search.DDGS", FakeDDGS)

    results = search_web("python")

    assert results == [
        SearchResult(title="Python", url="https://python.org", snippet="The Python language."),
        SearchResult(title="Docs", url="https://docs.python.org", snippet="Python documentation."),
    ]
