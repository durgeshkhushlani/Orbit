from dataclasses import dataclass

from ddgs import DDGS


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Run a text search and return the top results. Read-only, no
    confirmation needed -- searching carries no risk to the user's files."""
    raw_results = DDGS().text(query, max_results=max_results)
    return [
        SearchResult(title=r["title"], url=r["href"], snippet=r["body"]) for r in raw_results
    ]
