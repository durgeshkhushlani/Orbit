import chromadb
from chromadb.api.models.Collection import Collection

from orbit.config import settings


def get_collection() -> Collection:
    """Return the Orbit document collection, using ChromaDB's bundled default
    embedding function (ONNX MiniLM) unless a different one is wired in later."""
    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return client.get_or_create_collection(name=settings.chroma_collection_name)
