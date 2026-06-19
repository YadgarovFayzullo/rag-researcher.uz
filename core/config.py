import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Не задана обязательная переменная окружения: {name}")
    return value


class Settings:
    LLM_BASE_URL: str = _get("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_API_KEY: str = _get("LLM_API_KEY", "ollama")
    LLM_MODEL: str = _get("LLM_MODEL", "qwen3:8b")

    EMBED_BASE_URL: str = _get("EMBED_BASE_URL", LLM_BASE_URL)
    EMBED_API_KEY: str = _get("EMBED_API_KEY", LLM_API_KEY)
    EMBED_MODEL: str = _get("EMBED_MODEL", "nomic-embed-text")
    EMBED_DIM: int = int(_get("EMBED_DIM", "768"))

    QDRANT_URL: str | None = _get("QDRANT_URL")
    QDRANT_HOST: str = _get("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(_get("QDRANT_PORT", "6333"))
    QDRANT_API_KEY: str | None = _get("QDRANT_API_KEY")
    QDRANT_COLLECTION: str = _get("QDRANT_COLLECTION", "article_chunks")

    SUPABASE_URL: str | None = _get("SUPABASE_URL")
    SUPABASE_KEY: str | None = _get("SUPABASE_KEY")

    LANGFUSE_PUBLIC_KEY: str | None = _get("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY: str | None = _get("LANGFUSE_SECRET_KEY")
    LANGFUSE_BASE_URL: str | None = _get("LANGFUSE_BASE_URL")
    LANGFUSE_ENABLED: bool = _get("LANGFUSE_ENABLED", "true").lower() == "true"

    SEARCH_LIMIT: int = int(_get("SEARCH_LIMIT", "5"))
    SCORE_THRESHOLD: float = float(_get("SCORE_THRESHOLD", "0.5"))

    API_HOST: str = _get("API_HOST", "0.0.0.0")
    API_PORT: int = int(_get("API_PORT", "8000"))

    # CORS: список origin'ов через запятую, либо "*" для всех.
    CORS_ORIGINS: list[str] = [
        o.strip() for o in (_get("CORS_ORIGINS", "*") or "*").split(",") if o.strip()
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
