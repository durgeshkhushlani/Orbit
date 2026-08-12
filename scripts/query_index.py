"""Query Orbit's ChromaDB store directly, bypassing the LLM, to verify indexing.

Usage: python scripts/query_index.py "<query text>" [n_results]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbit.db.vectorstore import get_collection  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/query_index.py "<query text>" [n_results]')
        sys.exit(1)

    query = sys.argv[1]
    n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No results. Has anything been indexed yet?")
        return

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        snippet = doc[:300].replace("\n", " ") + ("..." if len(doc) > 300 else "")
        _print_safe(f"\n[{i}] distance={dist:.4f} source={meta.get('source')}")
        _print_safe(snippet)


def _print_safe(text: str) -> None:
    """Print text even if it contains characters the console encoding can't render."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


if __name__ == "__main__":
    main()
