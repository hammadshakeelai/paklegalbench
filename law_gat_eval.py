"""
Pakistan Law GAT (Graduate Assessment Test) & Bar Exam Benchmark Runner.

Evaluates statute retrieval and multi-choice legal reasoning across:
- Pakistan Penal Code (PPC 1860)
- Code of Criminal Procedure (CrPC 1898)
- Constitution of the Islamic Republic of Pakistan (1973)

Usage:
    python law_gat_eval.py
    python law_gat_eval.py --llm
    python law_gat_eval.py --out results/law_gat_report.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

from index import RetrievalConfig, Retriever, load_chunks
import llm

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "results" / "law_gat_eval.jsonl"
DEFAULT_OUT = ROOT / "results" / "law_gat_report.json"


def load_dataset() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing benchmark dataset: {DATASET_PATH}")
    return [json.loads(line) for line in DATASET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_retrieval(retriever: Retriever, questions: list[dict[str, Any]], cfg: RetrievalConfig, use_urdu: bool = False) -> dict[str, Any]:
    hits_at_1 = 0
    hits_at_5 = 0
    mrr_total = 0.0
    false_refusals = 0
    by_subject: dict[str, dict[str, Any]] = {}

    for q in questions:
        subj = q["subject"]
        if subj not in by_subject:
            by_subject[subj] = {"total": 0, "r1": 0, "r5": 0, "mrr": 0.0, "refusals": 0}
        by_subject[subj]["total"] += 1

        gold_ids = set(q["gold_ids"])
        q_text = (q.get("question_ur") if use_urdu else None) or q["question"]
        results = retriever.search(q_text, cfg)
        retrieved_ids = [h.chunk.id for h in results]

        is_refused = retriever.should_refuse(results, cfg, q_text)
        if is_refused:
            false_refusals += 1
            by_subject[subj]["refusals"] += 1

        # Recall@1
        if retrieved_ids and retrieved_ids[0] in gold_ids:
            hits_at_1 += 1
            by_subject[subj]["r1"] += 1

        # Recall@5
        if any(rid in gold_ids for rid in retrieved_ids[:5]):
            hits_at_5 += 1
            by_subject[subj]["r5"] += 1

        # MRR
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in gold_ids:
                rr = 1.0 / rank
                mrr_total += rr
                by_subject[subj]["mrr"] += rr
                break

    n = len(questions)
    overall = {
        "total_questions": n,
        "recall@1": round(hits_at_1 / n, 4) if n else 0,
        "recall@5": round(hits_at_5 / n, 4) if n else 0,
        "mrr": round(mrr_total / n, 4) if n else 0,
        "false_refusal_rate": round(false_refusals / n, 4) if n else 0,
    }

    subject_breakdown = {}
    for subj, stat in by_subject.items():
        st_n = stat["total"]
        subject_breakdown[subj] = {
            "n": st_n,
            "recall@1": round(stat["r1"] / st_n, 4) if st_n else 0,
            "recall@5": round(stat["r5"] / st_n, 4) if st_n else 0,
            "mrr": round(stat["mrr"] / st_n, 4) if st_n else 0,
            "false_refusals": stat["refusals"],
        }

    return {"overall": overall, "by_subject": subject_breakdown}


def evaluate_llm_answers(retriever: Retriever, questions: list[dict[str, Any]], cfg: RetrievalConfig, use_urdu: bool = False) -> dict[str, Any]:
    providers = llm.available_providers()
    if not providers:
        return {"error": "No LLM provider key configured"}

    correct = 0
    total = len(questions)
    details = []

    print(f"\nEvaluating LLM Answering on {total} Law GAT questions via {providers[0]}...")
    for idx, q in enumerate(questions, start=1):
        q_text = (q.get("question_ur") if use_urdu else None) or q["question"]
        results = retriever.search(q_text, cfg)
        prompt_q = (
            f"{q_text}\n\n"
            f"OPTIONS:\n"
            f"A) {q['options']['A']}\n"
            f"B) {q['options']['B']}\n"
            f"C) {q['options']['C']}\n"
            f"D) {q['options']['D']}\n\n"
            f"Which option (A, B, C, or D) is correct? Provide the letter and citation."
        )

        try:
            ans, prov = llm.answer(prompt_q, results)
            is_correct = f"({q['correct_option']})" in ans or f"Option {q['correct_option']}" in ans or ans.strip().startswith(q['correct_option'])
            if is_correct:
                correct += 1
            details.append({
                "id": q["id"],
                "question": q_text,
                "correct_option": q["correct_option"],
                "llm_answer": ans[:200],
                "is_correct": is_correct,
            })
            print(f"[{idx:02d}/{total}] {q['id']}: {'CORRECT' if is_correct else 'INCORRECT'}")
        except Exception as e:
            print(f"[{idx:02d}/{total}] {q['id']}: ERROR ({e})")
            details.append({"id": q["id"], "error": str(e), "is_correct": False})

    accuracy = round(correct / total, 4) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PakLegalBench on Law GAT & Bar Exam questions")
    parser.add_argument("--corpus", default=None, help="Path to corpus JSON (default chunks.json if present, else seed_corpus.json)")
    parser.add_argument("--urdu", action="store_true", help="Evaluate on authentic Urdu Law GAT bar exam questions")
    parser.add_argument("--llm", action="store_true", help="Run LLM question answering evaluation")
    parser.add_argument("--out", default=None, help="Output JSON report path")
    args = parser.parse_args()

    questions = load_dataset()
    edition_label = "Urdu Edition" if args.urdu else "English Edition"
    print(f"Loaded {len(questions)} Law GAT questions across PPC, CrPC, and Constitution ({edition_label}).")

    corpus_file = args.corpus or ("chunks.json" if Path("chunks.json").exists() else None)
    retriever = Retriever(load_chunks(corpus_file), load_dense=False)
    cfg = RetrievalConfig()

    t0 = time.perf_counter()
    retrieval_stats = evaluate_retrieval(retriever, questions, cfg, use_urdu=args.urdu)
    duration = time.perf_counter() - t0

    ov = retrieval_stats["overall"]
    header = f"PAKLEGALBENCH LAW GAT BENCHMARK RESULTS ({edition_label.upper()})"
    print("\n" + "=" * len(header))
    print(f"      {header}          ")
    print("=" * len(header))
    print(f"Questions Evaluated: {ov['total_questions']}")
    print(f"Recall@1:           {ov['recall@1'] * 100:.1f}%")
    print(f"Recall@5:           {ov['recall@5'] * 100:.1f}%")
    print(f"MRR:                {ov['mrr']:.4f}")
    print(f"False Refusal Rate: {ov['false_refusal_rate'] * 100:.1f}%")
    print(f"Evaluation Time:    {duration:.3f}s")
    print("\nBreakdown by Subject:")
    for subj, s in retrieval_stats["by_subject"].items():
        print(f"  [{subj.ljust(12)}] n={s['n']} | R@1: {s['recall@1']*100:.1f}% | R@5: {s['recall@5']*100:.1f}% | MRR: {s['mrr']:.4f} | False Refusals: {s['false_refusals']}")

    llm_stats = None
    if args.llm:
        llm_stats = evaluate_llm_answers(retriever, questions, cfg, use_urdu=args.urdu)
        if "accuracy" in llm_stats:
            print(f"\nLLM Exam Score: {llm_stats['correct']}/{llm_stats['total']} ({llm_stats['accuracy']*100:.1f}%)")

    default_output = ROOT / "results" / ("law_gat_urdu_report.json" if args.urdu else "law_gat_report.json")
    out_path = Path(args.out) if args.out else default_output
    report = {
        "benchmark": "Pakistan Law GAT (Bar Exam) Benchmark",
        "edition": "urdu" if args.urdu else "english",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrieval": retrieval_stats,
        "llm_answering": llm_stats,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
