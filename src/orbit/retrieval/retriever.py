from dataclasses import dataclass

from orbit.db.vectorstore import get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float


def retrieve(query: str, n_results: int = 5) -> list[RetrievedChunk]:
    """Return the top-k chunks most relevant to `query`."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        RetrievedChunk(text=doc, source=meta.get("source", "unknown"), distance=dist)
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
