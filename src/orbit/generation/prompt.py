from orbit.retrieval.retriever import RetrievedChunk

SYSTEM_INSTRUCTIONS = (
    "You are Orbit, a local assistant that answers questions using only the "
    "provided context. If the context doesn't contain the answer, say so "
    "plainly instead of guessing. Keep answers concise."
)


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble a grounded prompt from the user query and retrieved chunks."""
    if not chunks:
        context = "(no relevant context found)"
    else:
        context = "\n\n".join(
            f"[{i}] source: {chunk.source}\n{chunk.text}" for i, chunk in enumerate(chunks, 1)
        )

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
