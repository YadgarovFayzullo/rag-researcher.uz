from functools import lru_cache

from openai import OpenAI
from qdrant_client import QdrantClient

from core.config import settings


@lru_cache
def get_llm_client() -> OpenAI:
    return OpenAI(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)


@lru_cache
def get_embed_client() -> OpenAI:
    return OpenAI(base_url=settings.EMBED_BASE_URL, api_key=settings.EMBED_API_KEY)


@lru_cache
def get_qdrant() -> QdrantClient:
    if settings.QDRANT_URL:
        return QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    return QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )


@lru_cache
def get_supabase():
    from supabase import create_client

    if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY не заданы")
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


def get_embedding(text: str) -> list[float]:
    text = text[:8000]
    res = get_embed_client().embeddings.create(
        model=settings.EMBED_MODEL, input=text
    )
    return res.data[0].embedding
