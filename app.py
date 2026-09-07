"""
PakLegalBench MVP server.

    uvicorn app:app --reload

Endpoints:
    GET  /              chat UI
    POST /api/chat      {question, history?, config?} -> answer + sources
    GET  /api/health    corpus size, dense status, providers with keys
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

import llm
from index import RetrievalConfig, build_default

ROOT = Path(__file__).parent

app = FastAPI(title="PakLegalBench")

# Loading the embedding model takes ~10s. Set PLB_NO_DENSE=1 for fast restarts.
retriever = build_default() if not os.environ.get("PLB_NO_DENSE") else None
if retriever is None:
    from index import Retriever, load_chunks
    retriever = Retriever(load_chunks(), load_dense=False)


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []
    use_exact: bool = True
    use_sparse: bool = True
    use_dense: bool = True
    use_rerank: bool = False


@app.get("/api/health")
def health():
    return {
        "chunks": len(retriever.chunks),
        "dense": retriever.dense_available,
        "providers": llm.available_providers(),
        "corpus_verified": all(c.verified for c in retriever.chunks),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    cfg = RetrievalConfig(
        use_exact=req.use_exact,
        use_sparse=req.use_sparse,
        use_dense=req.use_dense,
        use_rerank=req.use_rerank,
    )
    hits = retriever.search(req.question, cfg)

    sources = [
        {
            "citation": h.chunk.citation(),
            "marginal_note": h.chunk.marginal_note,
            "act": h.chunk.act,
            "text": h.chunk.text,
            "score": round(h.score, 5),
            "channels": h.channels,
            "verified": h.chunk.verified,
            "source_url": h.chunk.source_url,
        }
        for h in hits
    ]

    if retriever.should_refuse(hits, cfg, req.question):
        return {
            "answer": (
                "I could not find a provision in the indexed corpus that "
                "addresses this. The corpus currently covers the Constitution, "
                "the PPC and the CrPC only, so the answer may exist elsewhere "
                "in Pakistani law."
            ),
            "refused": True,
            "sources": sources,
            "provider": None,
        }

    try:
        text, provider = llm.answer(req.question, hits, req.history)
    except llm.NoProviderError as e:
        return {
            "answer": f"Retrieval worked, generation is not configured. {e}",
            "refused": False,
            "sources": sources,
            "provider": None,
        }

    return {"answer": text, "refused": False, "sources": sources, "provider": provider}


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/research")
def research():
    return FileResponse(ROOT / "static" / "research.html")
