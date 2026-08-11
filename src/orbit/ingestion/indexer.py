import hashlib
from pathlib import Path

from orbit.db.vectorstore import get_collection
from orbit.ingestion.chunking import chunk_documents
from orbit.ingestion.loaders import load_folder


def _chunk_id(source: str, chunk_index: int) -> str:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"{digest}-{chunk_index}"


def index_folder(folder: Path) -> dict:
    """Load, chunk, and upsert every supported file under `folder` into Chroma.

    Upsert (not add) so re-running against the same folder updates existing
    chunks instead of erroring on duplicate IDs.
    """
    documents = load_folder(folder)
    chunks = chunk_documents(documents)

    if not chunks:
        return {"files_loaded": 0, "chunks_indexed": 0}

    ids = []
    texts = []
    metadatas = []
    source_counts: dict[str, int] = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        chunk_index = source_counts.get(source, 0)
        source_counts[source] = chunk_index + 1

        ids.append(_chunk_id(source, chunk_index))
        texts.append(chunk.page_content)
        metadatas.append({"source": source})

    collection = get_collection()
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    return {
        "files_loaded": len(source_counts),
        "chunks_indexed": len(chunks),
    }
