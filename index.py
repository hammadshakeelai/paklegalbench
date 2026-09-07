"""
Retrieval engine for PakLegalBench.

Three channels, fused with reciprocal rank fusion:
  1. exact  - regex-matched section/article references ("302 PPC", "Article 199")
  2. sparse - BM25 keyword matching
  3. dense  - embedding similarity (optional, needs sentence-transformers)

The exact channel exists because of the single most useful finding in the
research: ~31% of legal retrieval failures are requests for a specific article
reference, where dense models rank thematically-similar provisions above the
one actually named. Regex is a cheap, deterministic fix for that.

Everything degrades gracefully. No sentence-transformers -> sparse only.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).parent
DEFAULT_CORPUS = ROOT / "seed_corpus.json"

# Maps how people actually write act names to the act_short field.
ACT_ALIASES = {
    "ppc": "PPC",
    "p.p.c": "PPC",
    "p.p.c.": "PPC",
    "pakistan penal code": "PPC",
    "penal code": "PPC",
    "crpc": "CrPC",
    "cr.p.c": "CrPC",
    "cr.p.c.": "CrPC",
    "criminal procedure": "CrPC",
    "code of criminal procedure": "CrPC",
    "constitution": "Constitution",
    "article": "Constitution",
    "art": "Constitution",
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    act: str
    act_short: str
    year: int
    chapter: str
    section: str
    section_label: str
    marginal_note: str
    text: str
    jurisdiction: str
    status: str
    source_url: str
    verified: bool = False

    def citation(self) -> str:
        if self.act_short == "Constitution":
            return f"Constitution, {self.section_label}"
        return f"{self.act_short} {self.section_label}"

    def indexed_text(self) -> str:
        """
        What actually goes into the index. The prepended header is the
        '150-char summary' trick from the literature: a bare paragraph of a
        statute never says which act it belongs to, so both BM25 and the
        embedding model lose the one bit of context a lawyer needs.
        """
        return (
            f"{self.act_short} {self.section_label} {self.marginal_note}. "
            f"{self.chapter}. {self.text}"
        )


def load_chunks(path: Path | str | None = None) -> list[Chunk]:
    if path is None:
        path = os.environ.get("PLB_CORPUS_PATH", DEFAULT_CORPUS)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw["chunks"] if isinstance(raw, dict) else raw
    return [Chunk(**{k: v for k, v in r.items() if not k.startswith("_")}) for r in rows]


# --------------------------------------------------------------------------
# pure functions (these are the ones worth testing)
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

# Deliberately short. Legal queries are terse and over-stripping destroys the
# coverage signal used for refusal.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on",
    "at", "to", "for", "with", "and", "or", "if", "it", "its", "this", "that",
    "what", "which", "who", "whom", "how", "when", "where", "why", "can", "could",
    "do", "does", "did", "i", "my", "me", "you", "your", "under", "about", "any",
    "there", "s", "u", "sec", "please", "tell", "explain", "give",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps hyphenated forms like 489-f and 10a intact,
    which matters a lot for section numbers."""
    return TOKEN_RE.findall(text.lower())


REFERENCE_PATTERNS = [
    # "Article 199", "Art. 10A", "Article 10-A"
    re.compile(r"\b(?:article|art\.?)\s*([0-9]+(?:[-–]?[a-z])?)\b", re.I),
    # "Section 302 PPC", "s. 497 CrPC", "u/s 154"
    re.compile(r"\b(?:section|sec\.?|s\.|u/s)\s*([0-9]+(?:[-–]?[a-z])?)\b", re.I),
    # bare "302 PPC" / "497 CrPC" / "489-F P.P.C."
    re.compile(r"\b([0-9]+(?:[-–]?[a-z])?)\s*(?:of\s+the\s+)?(ppc|crpc|cr\.?p\.?c\.?|p\.?p\.?c\.?)\b", re.I),
]


def extract_references(query: str) -> list[tuple[str, str | None]]:
    """
    Pull explicit statutory references out of a query.
    Returns [(section, act_short_or_None), ...] with section normalised
    to uppercase, e.g. ("489-F", "PPC") or ("10A", "Constitution").
    """
    q = query.lower()
    act_hint = None
    for alias, short in ACT_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q):
            act_hint = short
            break

    found: list[tuple[str, str | None]] = []
    seen = set()
    for pat in REFERENCE_PATTERNS:
        for m in pat.finditer(query):
            raw_sec = m.group(1).upper()
            sec = re.sub(r"[–\s]", "-", raw_sec)
            act = act_hint
            if m.lastindex and m.lastindex >= 2:
                tail = (m.group(2) or "").lower().replace(".", "")
                act = "PPC" if tail == "ppc" else "CrPC"
            key = (sec, act)
            if key not in seen:
                seen.add(key)
                found.append(key)
    return found


def rrf(ranked_lists: list[list[str]], k: int = 60,
        weights: list[float] | None = None) -> list[tuple[str, float]]:
    """
    Reciprocal rank fusion. k=60 is the value the literature keeps landing on.

    Uses ranks rather than raw scores on purpose: BM25 scores, cosine
    similarities and cross-encoder scores are not comparable to each other,
    and cross-encoder scores are not even comparable across queries.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for lst, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


# --------------------------------------------------------------------------
# sparse
# --------------------------------------------------------------------------

class BM25:
    """Plain BM25-Okapi. k1 raised to 1.5 per the guidance for long-form legal
    text where term repetition carries real signal."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(self.N, 1)
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        for d in docs:
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t in counts:
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.N
        for term in query_tokens:
            if term not in self.df:
                continue
            idf = self._idf(term)
            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if not f:
                    continue
                dl = len(self.docs[i])
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


# --------------------------------------------------------------------------
# dense (optional)
# --------------------------------------------------------------------------

class DenseIndex:
    """Wraps sentence-transformers. Absent library -> available=False and the
    pipeline just runs without this channel."""

    def __init__(self, texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5"):
        self.available = False
        try:
            from sentence_transformers import SentenceTransformer  # noqa
            import numpy as np  # noqa
        except ImportError:
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.model = SentenceTransformer(model_name)
        emb = self.model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False)
        self.matrix = np.asarray(emb, dtype="float32")
        self.available = True

    def scores(self, query: str) -> list[float]:
        if not self.available:
            return []
        q = self.model.encode([query], normalize_embeddings=True)
        return (self.matrix @ self.np.asarray(q, dtype="float32").T).ravel().tolist()


class Reranker:
    """Cross-encoder. Also optional."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.available = False
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return
        self.model = CrossEncoder(model_name)
        self.available = True

    def rank(self, query: str, candidates: list[tuple[str, str]]) -> list[str]:
        """candidates: [(doc_id, text)] -> doc_ids best first."""
        if not self.available or not candidates:
            return [c[0] for c in candidates]
        scores = self.model.predict([(query, t) for _, t in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [candidates[i][0] for i in order]


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

@dataclass
class RetrievalConfig:
    use_exact: bool = True
    use_sparse: bool = True
    use_dense: bool = True
    use_rerank: bool = False       # off by default, it is slow to load
    candidates: int = 20
    top_k: int = 5
    rrf_k: int = 60
    weights: dict = field(default_factory=lambda: {"exact": 1.4, "sparse": 1.0, "dense": 0.8})

    # --- refusal ---
    # NOT a threshold on the fused score. RRF is rank-based, so the top fused
    # score is always 1/(k+1) regardless of how bad the match is. It carries no
    # relevance information at all and thresholding on it silently never fires.
    # Raw BM25 is no better: "Section 999 PPC" outscores real queries because
    # 'section', 'ppc' and 'penalty' all match something.
    #
    # Two deterministic signals instead:
    #   1. the query names a provision and the exact channel found no such
    #      provision  ->  refuse (kills false-premise traps)
    #   2. the top hit covers too few of the query's content words -> refuse
    refuse_on_missing_reference: bool = True
    min_term_coverage: float = 0.34


@dataclass
class Hit:
    chunk: Chunk
    score: float
    channels: list[str]


class Retriever:
    def __init__(self, chunks: list[Chunk], config: RetrievalConfig | None = None,
                 load_dense: bool = True):
        self.chunks = chunks
        self.config = config or RetrievalConfig()
        self.by_id = {c.id: c for c in chunks}
        self.ids = [c.id for c in chunks]
        texts = [c.indexed_text() for c in chunks]
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.dense = DenseIndex(texts) if load_dense else None
        self._reranker: Reranker | None = None

    @property
    def dense_available(self) -> bool:
        return bool(self.dense and self.dense.available)

    def _exact_channel(self, query: str) -> list[str]:
        refs = extract_references(query)
        if not refs:
            return []
        out = []
        for sec, act in refs:
            norm_sec = sec.replace("-", "").replace(" ", "").upper()
            for c in self.chunks:
                chunk_sec = c.section.replace("-", "").replace(" ", "").upper()
                if chunk_sec != norm_sec:
                    continue
                if act and c.act_short != act:
                    continue
                if c.id not in out:
                    out.append(c.id)
        return out

    def search(self, query: str, config: RetrievalConfig | None = None) -> list[Hit]:
        cfg = config or self.config
        lists: list[list[str]] = []
        weights: list[float] = []
        channel_of: dict[str, list[str]] = {}

        def add(name: str, ids: list[str]):
            if not ids:
                return
            lists.append(ids)
            weights.append(cfg.weights.get(name, 1.0))
            for i in ids:
                channel_of.setdefault(i, []).append(name)

        if cfg.use_exact:
            add("exact", self._exact_channel(query))

        if cfg.use_sparse:
            s = self.bm25.scores(tokenize(query))
            ranked = sorted(range(len(s)), key=lambda i: -s[i])[:cfg.candidates]
            add("sparse", [self.ids[i] for i in ranked if s[i] > 0])

        if cfg.use_dense and self.dense_available:
            s = self.dense.scores(query)
            ranked = sorted(range(len(s)), key=lambda i: -s[i])[:cfg.candidates]
            add("dense", [self.ids[i] for i in ranked])

        if not lists:
            return []

        fused = rrf(lists, k=cfg.rrf_k, weights=weights)[:cfg.candidates]

        if cfg.use_rerank:
            if self._reranker is None:
                self._reranker = Reranker()
            order = self._reranker.rank(
                query, [(i, self.by_id[i].indexed_text()) for i, _ in fused]
            )
            score_map = dict(fused)
            fused = [(i, score_map[i]) for i in order]

        return [
            Hit(chunk=self.by_id[i], score=sc, channels=channel_of.get(i, []))
            for i, sc in fused[:cfg.top_k]
        ]

    def refusal_reason(self, query: str, hits: list[Hit],
                       config: RetrievalConfig | None = None) -> str | None:
        """None means answer. A string means refuse, and says why."""
        cfg = config or self.config

        if not hits:
            return "nothing_retrieved"

        # 1. named a provision we do not have
        if cfg.refuse_on_missing_reference:
            refs = extract_references(query)
            if refs and not self._exact_channel(query):
                labels = ", ".join(
                    f"{a or ''} {s}".strip() for s, a in refs
                )
                return f"unknown_provision:{labels}"

        # 2. top hit barely overlaps the question's content words
        # If the exact channel matched the provision and it is the top hit, don't false-refuse
        # on lexical term coverage (e.g. "u/s 497 Cr.P.C." where abbreviation dots dilute word overlap).
        if "exact" in hits[0].channels:
            return None

        q_terms = {t for t in tokenize(query) if t not in STOPWORDS}
        if q_terms:
            top_terms = set(tokenize(hits[0].chunk.indexed_text()))
            coverage = len(q_terms & top_terms) / len(q_terms)
            if coverage < cfg.min_term_coverage:
                return f"low_coverage:{coverage:.2f}"

        return None

    def should_refuse(self, hits: list[Hit], config: RetrievalConfig | None = None,
                      query: str = "") -> bool:
        return self.refusal_reason(query, hits, config) is not None


def build_default() -> Retriever:
    return Retriever(load_chunks())


if __name__ == "__main__":
    import sys

    r = build_default()
    print(f"{len(r.chunks)} chunks. dense={'on' if r.dense_available else 'off (sparse only)'}\n")
    q = " ".join(sys.argv[1:]) or "what is the punishment for cheating"
    print(f"query: {q}\n")
    for h in r.search(q):
        print(f"  {h.score:.4f}  [{'+'.join(h.channels)}]  {h.chunk.citation()} — {h.chunk.marginal_note}")
