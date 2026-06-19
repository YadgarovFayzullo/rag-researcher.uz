from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.clients import get_qdrant
from core.config import settings
from core.ingest import delete_tenant, index_posts
from core.rag import ask, search_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_qdrant().get_collections()
    except Exception as e:
        print(f"[startup] Qdrant пока недоступен: {e}")
    yield


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)

# CORS: разрешённые origin'ы берём из настроек (CORS_ORIGINS в .env).
# Если "*" — нельзя одновременно отдавать credentials по спецификации CORS.
_allow_all = settings.CORS_ORIGINS == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    limit: int | None = Field(None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[str | None]


class SearchRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    limit: int | None = Field(None, ge=1, le=20)
    section: str | None = None
    is_reference: bool | None = None  # None=все, True=только референс, False=только свой


class PostItem(BaseModel):
    # int — message_id основного канала; str — префиксованный id референс-канала
    # (напр. "@tech:41"), чтобы не было коллизий point-id в Qdrant.
    id: int | str
    text: str
    date: str | None = None


class IndexRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    posts: list[PostItem] = Field(..., min_length=1)
    is_reference: bool = False  # True — посты референс-канала


class IndexResponse(BaseModel):
    indexed: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    try:
        get_qdrant().get_collections()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"qdrant unavailable: {e}")


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    try:
        answer, sources = ask(req.question, req.tenant_id, limit=req.limit)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
def index_endpoint(req: IndexRequest):
    try:
        n = index_posts(
            req.tenant_id, [p.model_dump() for p in req.posts], req.is_reference
        )
        return IndexResponse(indexed=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)


@app.post("/delete")
def delete_endpoint(req: DeleteRequest):
    try:
        delete_tenant(req.tenant_id)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_endpoint(req: SearchRequest):
    try:
        results = search_chunks(
            req.query,
            req.tenant_id,
            limit=req.limit,
            section=req.section,
            is_reference=req.is_reference,
        )
        return {
            "results": [
                {
                    "score": r.score,
                    "section": r.payload.get("section"),
                    "title": r.payload.get("title"),
                    "content": r.payload.get("content"),
                    "article_id": r.payload.get("article_id"),
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
