from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.clients import get_qdrant
from core.config import settings
from core.rag import ask, search_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_qdrant().get_collections()
    except Exception as e:
        print(f"[startup] Qdrant пока недоступен: {e}")
    yield


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    limit: int | None = Field(None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[str | None]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int | None = Field(None, ge=1, le=20)
    section: str | None = None


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
        answer, sources = ask(req.question, limit=req.limit)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search_endpoint(req: SearchRequest):
    try:
        results = search_chunks(req.query, limit=req.limit, section=req.section)
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
