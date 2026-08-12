from dataclasses import dataclass

from orbit.generation.prompt import build_prompt
from orbit.llm.ollama_client import generate
from orbit.retrieval.retriever import retrieve


@dataclass
class RagAnswer:
    answer: str
    sources: list[str]


def answer_query(query: str, n_results: int = 5) -> RagAnswer:
    """Retrieve relevant chunks, ground a prompt in them, and generate an answer."""
    chunks = retrieve(query, n_results=n_results)
    prompt = build_prompt(query, chunks)
    answer = generate(prompt)

    sources = list(dict.fromkeys(chunk.source for chunk in chunks))
    return RagAnswer(answer=answer, sources=sources)
