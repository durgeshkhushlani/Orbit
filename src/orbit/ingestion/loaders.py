from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


def load_document(path: Path) -> list[Document]:
    """Load a single file into LangChain Documents, dispatching by extension."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(str(path)).load()

    if suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        return Docx2txtLoader(str(path)).load()

    if suffix in (".md", ".txt"):
        from langchain_community.document_loaders import TextLoader

        return TextLoader(str(path), encoding="utf-8").load()

    raise ValueError(f"Unsupported file type: {suffix}")


def load_folder(folder: Path) -> list[Document]:
    """Recursively load every supported file under a folder."""
    documents: list[Document] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.extend(load_document(path))
    return documents
