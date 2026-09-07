"""
PakLegalBench MVP server.

    uvicorn app:app --reload

Endpoints:
    GET  /              chat UI
    POST /api/chat      {question, history?, config?} -> answer + sources
    GET  /api/health    corpus size, dense status, providers with keys
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import llm
from index import RetrievalConfig, build_default

ROOT = Path(__file__).parent

app = FastAPI(title="PakLegalBench")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# Loading the embedding model takes ~10s. Set PLB_NO_DENSE=1 for fast restarts.
default_corpus = ROOT / "chunks.json" if (ROOT / "chunks.json").exists() else ROOT / "seed_corpus.json"
corpus_path = os.environ.get("PLB_CORPUS_PATH", str(default_corpus))
retriever = build_default(corpus_path) if not os.environ.get("PLB_NO_DENSE") else None
if retriever is None:
    from index import Retriever, load_chunks
    retriever = Retriever(load_chunks(corpus_path), load_dense=False)


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=4000)
    history: list[dict] = Field(default_factory=list, max_length=10)
    use_exact: bool = True
    use_sparse: bool = True
    use_dense: bool = True
    use_rerank: bool = False


class HealthResponse(BaseModel):
    chunks: int
    dense: bool
    providers: list[str]
    corpus_verified: bool


class SourceItem(BaseModel):
    citation: str
    marginal_note: str
    act: str
    text: str
    score: float
    channels: list[str]
    verified: bool
    source_url: str
    cross_references: list[str] = []


class ChatResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[SourceItem] = []
    provider: str | None = None


@app.get("/api/health", response_model=HealthResponse)
def health():
    return {
        "chunks": len(retriever.chunks),
        "dense": retriever.dense_available,
        "providers": llm.available_providers(),
        "corpus_verified": all(c.verified for c in retriever.chunks),
    }


GREETINGS = {"hi", "hello", "hey", "salam", "assalam o alaikum", "aoa", "help", "start"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q_norm = req.question.strip().lower().rstrip("!?.,")
    if q_norm in GREETINGS:
        return {
            "answer": (
                "Hello! I am PakLegalBench, an AI statute retrieval assistant for Pakistani law.\n\n"
                "I answer questions strictly grounded in the **Constitution of Pakistan (1973)**, "
                "the **Pakistan Penal Code (PPC 1860)**, and the **Code of Criminal Procedure (CrPC 1898)** with citations.\n\n"
                "Try asking:\n"
                "• *What is the punishment under Section 420 PPC?*\n"
                "• *When can bail be granted in a non-bailable offence?*\n"
                "• *What is the difference between 302 and 320 PPC?*\n"
                "• *Article 199 writ jurisdiction*"
            ),
            "refused": False,
            "sources": [],
            "provider": "assistant",
        }

    cfg = RetrievalConfig(
        use_exact=req.use_exact,
        use_sparse=req.use_sparse,
        use_dense=req.use_dense,
        use_rerank=req.use_rerank,
    )
    hits = retriever.search(req.question, cfg)

    graph = llm._get_statute_graph()
    sources = []
    for h in hits:
        c = h.chunk
        sec_norm = c.section.replace("-", "").replace(" ", "").upper()
        key = f"{c.act_short}:{sec_norm}"
        refs = graph.get(key, [])
        sources.append({
            "citation": c.citation(),
            "marginal_note": c.marginal_note,
            "act": c.act,
            "text": c.text,
            "score": round(h.score, 5),
            "channels": h.channels,
            "verified": c.verified,
            "source_url": c.source_url,
            "cross_references": refs,
        })

    if retriever.should_refuse(hits, cfg, req.question):
        reason = retriever.refusal_reason(req.question, hits, cfg)
        if reason and reason.startswith("foreign_jurisdiction:"):
            act = reason.split(":", 1)[1]
            msg = (
                f"The requested provision belongs to {act}, which is outside Pakistani jurisdiction. "
                "PakLegalBench exclusively indexes Pakistani federal statutes (the Constitution 1973, "
                "the Pakistan Penal Code 1860, and the Code of Criminal Procedure 1898)."
            )
        elif reason and reason.startswith("unknown_provision:"):
            prov = reason.split(":", 1)[1]
            msg = (
                f"I could not find {prov} in the indexed corpus. "
                "The corpus currently covers the Constitution (1973), PPC (1860), and CrPC (1898) only. "
                "Please verify the provision number."
            )
        else:
            msg = (
                "I could not find a provision in the indexed corpus that addresses this. "
                "The system currently indexes the Constitution, the PPC, and the CrPC only. "
                "Out-of-scope topics (e.g., civil service pay scales / BPS grades, tax ordinances, corporate rules) "
                "are intentionally refused to prevent hallucination."
            )
        return {
            "answer": msg,
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


@app.get("/api/benchmarks")
def benchmarks():
    data = {}
    files = {
        "retrieval": ROOT / "results" / "retrieval.json",
        "law_gat": ROOT / "results" / "law_gat_report.json",
        "stanford": ROOT / "results" / "stanford_eval_report.json",
        "redteam": ROOT / "results" / "redteam_report.json",
        "legal_uqa_eng": ROOT / "results" / "legal_uqa_eng.json",
        "legal_uqa_urdu": ROOT / "results" / "legal_uqa_urdu.json",
    }
    for k, p in files.items():
        if p.exists():
            try:
                data[k] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                data[k] = None
    return data


@app.get("/api/graph")
def get_graph(provision: str | None = None):
    """
    Returns statutory reference graph.
    If ?provision=PPC:302 is passed, returns adjacency for that provision.
    Otherwise returns graph summary metrics.
    """
    graph = llm._get_statute_graph()
    if provision:
        p_clean = provision.upper().replace("-", "").strip()
        edges = graph.get(p_clean, [])
        inbound = [src for src, dsts in graph.items() if p_clean in dsts]
        return {
            "provision": p_clean,
            "outbound_citations": edges,
            "inbound_citations": inbound,
        }
    return {
        "total_provisions_with_edges": len(graph),
        "total_directed_edges": sum(len(dsts) for dsts in graph.values()),
        "sample_nodes": list(graph.keys())[:15],
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/research")
def research():
    return FileResponse(ROOT / "static" / "research.html")
