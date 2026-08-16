import trafilatura


def extract_content(url: str) -> str | None:
    """Fetch `url` and pull its main readable text, discarding boilerplate
    (nav, ads, comments). Returns None if the page can't be fetched or
    nothing extractable is found -- read-only, no confirmation needed."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        return None
    return trafilatura.extract(downloaded, include_comments=False, include_tables=False)
