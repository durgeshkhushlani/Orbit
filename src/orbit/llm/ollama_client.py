import httpx

from orbit.config import settings


def check_ollama() -> dict:
    """Check that Ollama is reachable and the configured model is pulled."""
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"reachable": False, "model_available": False, "error": str(exc)}

    models = [m["name"] for m in response.json().get("models", [])]
    return {
        "reachable": True,
        "model_available": settings.ollama_model in models,
        "models": models,
    }


def generate(prompt: str, timeout: float = 120.0) -> str:
    """Send a prompt to the configured Ollama model and return its response text."""
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]
