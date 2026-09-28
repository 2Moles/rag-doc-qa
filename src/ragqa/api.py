"""HTTP API. Run: uvicorn ragqa.api:app --reload  (documents from $RAGQA_DOCS, default data/sample_docs)"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .answer import answer, claude_available
from .documents import chunk_pages, load_folder
from .retrieval import HybridIndex

app = FastAPI(title="rag-doc-qa", version="1.0.0")


@lru_cache(maxsize=1)
def get_index() -> HybridIndex:
    folder = Path(os.environ.get("RAGQA_DOCS", "data/sample_docs"))
    return HybridIndex(chunk_pages(load_folder(folder)))


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)
    mode: str = Field(default="hybrid", pattern="^(bm25|tfidf|hybrid)$")


@app.get("/health")
def health() -> dict:
    index = get_index()
    mode = "claude" if claude_available() else "extractive"
    return {"status": "ok", "chunks": len(index.chunks), "answer_mode": mode}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    hits = get_index().search(req.question, k=req.k, mode=req.mode)
    try:
        result = answer(req.question, hits)
    except Exception as exc:  # surface API/network failures as 502, not a stack trace
        raise HTTPException(status_code=502, detail=f"answer generation failed: {exc}") from exc
    return {"question": req.question, "answer": result.text, "answer_mode": result.mode, "sources": result.sources}
