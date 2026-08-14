from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection_name: str = "orbit_documents"

    checkpoint_db_path: Path = Path("./data/checkpoints.db")
    retrieval_confidence_threshold: float = 1.0

    orbit_allowed_dirs: str = ""

    @property
    def allowed_dirs(self) -> list[Path]:
        """Root directories the File Agent may touch, resolved to absolute paths.
        Configured as a comma-separated list in ORBIT_ALLOWED_DIRS."""
        return [Path(raw.strip()).resolve() for raw in self.orbit_allowed_dirs.split(",") if raw.strip()]


settings = Settings()
