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


def load_legal_uqa() -> list[dict]:
    """LEGAL-UQA gives (question, answer, context-article). Mapping its article
    labels onto our chunk ids is the step that decides whether every number
    below means anything. Verify 30 pairs by hand before trusting it."""
    from datasets import load_dataset
    ds = load_dataset("faizanfaisal/legal-uqa", split="train")
    chunks = load_chunks()
    by_section = {}
    for c in chunks:
        if c.act_short == "Constitution":
            by_section.setdefault(c.section.upper(), []).append(c.id)

    cases, unmapped = [], 0
    for row in ds:
        import re
        m = re.search(r"\b(?:article)\s*([0-9]+[A-Za-z]?)", str(row.get("context", "")), re.I)
        if not m:
            unmapped += 1
            continue
        ids = by_section.get(m.group(1).upper())
        if not ids:
            unmapped += 1
            continue
        cases.append({"question": row["question"], "gold_ids": ids})
    print(f"mapped {len(cases)} cases, {unmapped} unmapped\n")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adversarial", action="store_true")
    ap.add_argument("--legal-uqa", action="store_true")
    ap.add_argument("--out", default="results/retrieval.json")
    args = ap.parse_args()

    cases = []
    if args.adversarial or not args.legal_uqa:
        cases += load_adversarial()
    if args.legal_uqa:
        cases += load_legal_uqa()

    retriever = Retriever(load_chunks())
    print(f"{len(cases)} cases · dense {'on' if retriever.dense_available else 'OFF'}\n")

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
