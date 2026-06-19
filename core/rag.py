from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.clients import get_embedding, get_qdrant
from core.config import settings

SYSTEM_PROMPT = (
    "Отвечай только на основе контекста. Если ответа нет — скажи что не знаешь.\n\n"
    "Контекст:\n{context}"
)


def _chat_client():
    if settings.LANGFUSE_ENABLED and settings.LANGFUSE_SECRET_KEY:
        try:
            from langfuse.openai import OpenAI as TracedOpenAI

            return TracedOpenAI(
                base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY
            )
        except Exception:
            pass
    from core.clients import get_llm_client

    return get_llm_client()


def _build_filter(
    tenant_id: str,
    section: str | None = None,
    is_reference: bool | None = None,
) -> Filter:
    """Жёсткая изоляция арендатора: tenant_id обязателен в каждом поиске.

    is_reference=None — без фильтра (свои + референс); True/False — только
    референс / только свой канал (для квотированного retrieval).
    """
    must = [FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    if section:
        must.append(FieldCondition(key="section", match=MatchValue(value=section)))
    if is_reference is not None:
        must.append(
            FieldCondition(key="is_reference", match=MatchValue(value=is_reference))
        )
    return Filter(must=must)


def search_chunks(
    query: str,
    tenant_id: str,
    limit: int | None = None,
    section: str | None = None,
    is_reference: bool | None = None,
):
    vector = get_embedding(query)

    results = get_qdrant().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vector,
        limit=limit or settings.SEARCH_LIMIT,
        score_threshold=settings.SCORE_THRESHOLD,
        query_filter=_build_filter(tenant_id, section, is_reference),
    ).points
    return results


def ask(
    question: str, tenant_id: str, limit: int | None = None
) -> tuple[str, list[str]]:
    chunks = search_chunks(question, tenant_id, limit=limit)
    context = [c.payload["content"] for c in chunks]
    sources = [c.payload.get("title") for c in chunks]
    context_str = "\n\n".join(context)

    response = _chat_client().chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context_str)},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content, sources
