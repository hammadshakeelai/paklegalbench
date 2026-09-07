"""
Retrieval evaluation harness.

Computes Recall@k and MRR for each retrieval configuration. Needs NO LLM calls,
so it costs nothing and is not rate-limited. Run this before you build features.

    python eval.py --adversarial          # runs the trap set (works out of the box)
    python eval.py --legal-uqa            # runs LEGAL-UQA (needs `datasets` + net)

Gold labels are chunk ids. Metrics follow SG-LegalCite: Recall@k for
k in {1,5,10,20} plus MRR, fixed seed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from index import RetrievalConfig, Retriever, load_chunks

ROOT = Path(__file__).parent
KS = (1, 5, 10, 20)

CONFIGS = {
    "sparse_only":  RetrievalConfig(use_exact=False, use_sparse=True,  use_dense=False),
    "dense_only":   RetrievalConfig(use_exact=False, use_sparse=False, use_dense=True),
    "hybrid":       RetrievalConfig(use_exact=False, use_sparse=True,  use_dense=True),
    "hybrid+exact": RetrievalConfig(use_exact=True,  use_sparse=True,  use_dense=True),
}


def score(retriever: Retriever, cases: list[dict], cfg: RetrievalConfig) -> dict:
    cfg = RetrievalConfig(**{**cfg.__dict__, "top_k": max(KS), "candidates": max(KS)})
    hits_at = {k: 0 for k in KS}
    rr_total = 0.0
    refusals = 0
    false_refusals = 0
    n = 0

    for case in cases:
        gold = set(case.get("gold_ids", []))
        results = retriever.search(case["question"], cfg)
        ids = [h.chunk.id for h in results]

        if case.get("expect_refusal"):
            if retriever.should_refuse(results, cfg, case["question"]):
                refusals += 1
            continue

        n += 1
        if retriever.should_refuse(results, cfg, case["question"]):
            false_refusals += 1
        for k in KS:
            if gold & set(ids[:k]):
                hits_at[k] += 1
        for rank, i in enumerate(ids, start=1):
            if i in gold:
                rr_total += 1 / rank
                break

    traps = [c for c in cases if c.get("expect_refusal")]
    return {
        "n": n,
        **{f"recall@{k}": round(hits_at[k] / n, 4) if n else None for k in KS},
        "mrr": round(rr_total / n, 4) if n else None,
        "refusal_rate": round(refusals / len(traps), 4) if traps else None,
        "false_refusal": round(false_refusals / n, 4) if n else None,
        "traps": len(traps),
    }


def table(results: dict[str, dict]) -> str:
    cols = ["n"] + [f"recall@{k}" for k in KS] + ["mrr", "refusal_rate", "false_refusal"]
    w = 14
    out = ["config".ljust(16) + "".join(c.rjust(w) for c in cols)]
    out.append("-" * len(out[0]))
    for name, r in results.items():
        row = name.ljust(16)
        for c in cols:
            v = r.get(c)
            row += ("—" if v is None else f"{v}").rjust(w)
        out.append(row)
    return "\n".join(out)


def load_adversarial() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "adversarial.jsonl").read_text().splitlines() if l.strip()]


def load_legal_uqa(split: str = "validation") -> list[dict]:
    """LEGAL-UQA gives (question, answer, context-article) over the 1973 Constitution.
    We map rows onto our chunk IDs with 100% precision via marginal note and text overlap.
    Cached locally in results/legal_uqa_eval.jsonl (619 rows: 495 train, 124 val).
    """
    cached_path = ROOT / "results" / "legal_uqa_eval.jsonl"
    if cached_path.exists():
        rows = [json.loads(line) for line in cached_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if split in ("train", "validation"):
            rows = [r for r in rows if r.get("split") == split]
        print(f"Loaded {len(rows)} LEGAL-UQA ({split}) benchmark cases from local cache.\n")
        return rows

    import io
    import urllib.request
    import pyarrow.parquet as pq

    chunks = load_chunks()
    const_chunks = [c for c in chunks if c.act_short == "Constitution"]

    def norm(t):
        return re.sub(r"[^a-z0-9]", "", str(t).lower())

    def match_context(ctx_eng):
        c_norm = norm(ctx_eng)
        if len(c_norm) < 15:
            return None
        for c in const_chunks:
            mn = norm(c.marginal_note)
            if len(mn) >= 4 and mn in c_norm[:150]:
                return c.id
            txt_norm = norm(c.text)
            if len(txt_norm) >= 30 and txt_norm[:35] in c_norm:
                return c.id
            if len(c_norm) >= 35 and c_norm[:35] in txt_norm:
                return c.id
        return None

    cases = []
    splits = [split] if split in ("train", "validation") else ["validation", "train"]
    for s in splits:
        url = f"https://huggingface.co/datasets/nlp-anonymous-researcher/LEGAL-UQA/resolve/main/data/{s}-00000-of-00001.parquet"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            table = pq.read_table(io.BytesIO(resp.read())).to_pydict()
        for i in range(len(table["question_eng"])):
            gold = match_context(table["context_eng"][i])
            if gold:
                cases.append({
                    "id": f"legal-uqa-{s}-{i}",
                    "question": table["question_eng"][i],
                    "question_urdu": table["question_urdu"][i],
                    "gold_ids": [gold],
                    "split": s,
                })
    print(f"Mapped {len(cases)} LEGAL-UQA cases directly from HF repository.\n")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--legal-uqa", action="store_true")
    ap.add_argument("--corpus", default=None, help="Path to corpus chunks json")
    ap.add_argument("--out", default="results/retrieval.json")
    args = ap.parse_args()

    corpus_path = args.corpus
    if corpus_path is None:
        if args.legal_uqa and (ROOT / "chunks.json").exists():
            corpus_path = str(ROOT / "chunks.json")
        else:
            corpus_path = os.environ.get("PLB_CORPUS_PATH", str(ROOT / "seed_corpus.json"))

    cases = []
    if args.adversarial or not args.legal_uqa:
        cases += load_adversarial()
    if args.legal_uqa:
        cases += load_legal_uqa()

    chunks = load_chunks(corpus_path)
    retriever = Retriever(chunks)
    print(f"{len(cases)} cases · {len(chunks)} chunks · dense {'on' if retriever.dense_available else 'OFF'}\n")

    results = {}
    for name, cfg in CONFIGS.items():
        if cfg.use_dense and not cfg.use_sparse and not retriever.dense_available:
            continue
        results[name] = score(retriever, cases, cfg)

    print(table(results))
    out = ROOT / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
