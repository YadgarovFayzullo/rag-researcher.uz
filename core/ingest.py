"""Индексация постов Telegram-канала в Qdrant (мульти-тенант).

Используется новым эндпоинтом POST /index: бот (content-ai) скрейпит историю
своего канала и шлёт сюда посты пачкой вместе со своим tenant_id. Изоляция
арендаторов — через payload.tenant_id (одна общая коллекция, не таблица на канал).
"""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from core.clients import get_embedding, get_qdrant
from core.config import settings

COLLECTION = settings.QDRANT_COLLECTION

# Стабильное пространство имён, чтобы (tenant_id, post_id) давал детерминированный
# id точки — повторная индексация того же поста обновляет, а не дублирует.
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def ensure_collection() -> None:
    q = get_qdrant()
    if not q.collection_exists(COLLECTION):
        q.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=settings.EMBED_DIM, distance=Distance.COSINE
            ),
        )
        # Индекс по tenant_id ускоряет фильтрацию и обязателен для надёжной изоляции.
        q.create_payload_index(COLLECTION, "tenant_id", PayloadSchemaType.KEYWORD)
        # Флаг референс-канала — для квотированного retrieval (свой канал vs источники).
        q.create_payload_index(COLLECTION, "is_reference", PayloadSchemaType.BOOL)


def _point_id(tenant_id: str, post_id: Any) -> str:
    return str(uuid.uuid5(_NS, f"{tenant_id}:{post_id}"))


def _chunk_text(text: str, max_chunk_size: int = 1500, min_chunk_size: int = 20) -> list[str]:
    """Режет длинный текст на чанки по абзацам, стараясь не превышать max_chunk_size.

    Пропускает чанки короче min_chunk_size (noise filter).
    """
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return [text] if len(text) >= min_chunk_size else []

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)
        # Если уже в чанке что-то есть и добавление этого абзаца превышит лимит:
        if current_chunk and current_size + para_size + 1 > max_chunk_size:
            chunk_text = '\n'.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size + 1

    # Последний чанк
    if current_chunk:
        chunk_text = '\n'.join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)

    return chunks if chunks else []


def index_posts(
    tenant_id: str, posts: list[dict[str, Any]], is_reference: bool = False
) -> int:
    """Индексирует посты одного арендатора. Возвращает число загруженных чанков.

    Каждый post: {"id": <int|str>, "text": <str>, "date": <str|None>}.
    is_reference=True — посты стороннего (референс) канала: участвуют в retrieval,
    но через отдельную квоту, чтобы не вытесняться постами своего канала.

    Фильтрация:
    - пропускает пусто и короче 20 символов (noise)
    - чанкирует длинные посты (>1500 символов) по абзацам, каждый чанк индексируется отдельно
    """
    ensure_collection()
    q = get_qdrant()

    points: list[PointStruct] = []
    for p in posts:
        text = (p.get("text") or "").strip()
        if not text or len(text) < 20:
            continue

        # Чанкируем текст (или берём целиком, если он короче лимита)
        chunks = _chunk_text(text, max_chunk_size=1500, min_chunk_size=20)

        for chunk_idx, chunk in enumerate(chunks):
            # Для чанков используем composite ID: post_id:chunk_index
            chunk_id = f"{p['id']}:{chunk_idx}" if len(chunks) > 1 else p['id']
            points.append(
                PointStruct(
                    id=_point_id(tenant_id, chunk_id),
                    vector=get_embedding(chunk),
                    payload={
                        "tenant_id": tenant_id,
                        "post_id": p["id"],
                        "is_reference": is_reference,
                        "title": None,
                        "content": chunk[:2000],
                        "date": p.get("date"),
                    },
                )
            )

    for i in range(0, len(points), 100):
        q.upsert(collection_name=COLLECTION, points=points[i : i + 100])

    return len(points)


def delete_tenant(tenant_id: str) -> None:
    """Удаляет все вектора арендатора (вызывается при удалении канала)."""
    q = get_qdrant()
    if not q.collection_exists(COLLECTION):
        return
    q.delete(
        collection_name=COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
            )
        ),
    )
