"""Index every supported file under a folder into Orbit's ChromaDB store.

Usage: python scripts/index_folder.py <folder>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orbit.ingestion.indexer import index_folder  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/index_folder.py <folder>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    summary = index_folder(folder)
    print(f"Files loaded:   {summary['files_loaded']}")
    print(f"Chunks indexed: {summary['chunks_indexed']}")


if __name__ == "__main__":
    main()
