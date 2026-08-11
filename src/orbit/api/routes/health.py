from fastapi import APIRouter

from orbit.llm.ollama_client import check_ollama

router = APIRouter()


@router.get("/health")
def health() -> dict:
    ollama_status = check_ollama()
    return {
        "status": "ok" if ollama_status["reachable"] else "degraded",
        "ollama": ollama_status,
    }
