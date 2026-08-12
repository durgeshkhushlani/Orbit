"""Ask Orbit a question against the indexed documents: retrieve + generate.

Usage: python scripts/ask.py "<question>" [n_results]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbit.rag.pipeline import answer_query  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python scripts/ask.py "<question>" [n_results]')
        sys.exit(1)

    query = sys.argv[1]
    n_results = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    result = answer_query(query, n_results=n_results)

    print(result.answer.strip())
    print("\nSources:")
    for source in result.sources:
        print(f"  - {source}")


if __name__ == "__main__":
    main()
