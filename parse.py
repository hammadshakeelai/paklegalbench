"""
Build the real corpus. Replaces seed_corpus.json.

Day 1 of the plan, and the day that decides whether every number the eval
harness prints means anything. Do not rush it.

    python parse.py --out chunks.json

The section regexes below are a starting point, not a finished parser.
Pakistani acts are inconsistently formatted and the amendment footnotes are the
hard part. Expect to iterate. The rule: dump 20 random chunks and read them.
If any looks wrong, fix the parser before moving on.
"""
from __future__ import annotations

import argparse
import json
import re

# Matches "302. Punishment of qatl-i-amd." and "489-F. Dishonestly issuing..."
SECTION_RE = re.compile(
    r"^\s*(?P<num>\d+[-–]?[A-Z]{0,2})\.\s+(?P<note>[^\n.]{3,120})\.",
    re.M,
)
ARTICLE_RE = re.compile(
    r"^\s*(?P<num>\d+[A-Z]?)\.\s+(?P<note>[^\n.]{3,120})\.",
    re.M,
)

# Flag these for manual review rather than silently guessing status.
AMENDMENT_HINTS = re.compile(
    r"\b(omitted|substituted|inserted|repealed|added by|as amended)\b", re.I
)


def split_sections(text: str, pattern: re.Pattern) -> list[dict]:
    matches = list(pattern.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        out.append({
            "section": m.group("num").replace("–", "-").upper(),
            "marginal_note": m.group("note").strip(),
            "text": re.sub(r"\s+", " ", body),
            "needs_review": bool(AMENDMENT_HINTS.search(body)),
        })
    return out


def from_hf_pakistan_laws(target_titles: list[str]) -> list[dict]:
    """AyeshaJadoon/Pakistan_Laws_Dataset: 969 acts as {file_name, content}."""
    from datasets import load_dataset
    ds = load_dataset("AyeshaJadoon/Pakistan_Laws_Dataset", split="train")
    wanted = [r for r in ds
              if any(t.lower() in str(r["file_name"]).lower() for t in target_titles)]
    print(f"matched {len(wanted)} of {len(ds)} documents")
    return wanted


ACTS = {
    "constitution": dict(act="Constitution of the Islamic Republic of Pakistan, 1973",
                         act_short="Constitution", year=1973, pattern=ARTICLE_RE,
                         label=lambda n: f"Article {n}"),
    "penal": dict(act="Pakistan Penal Code, 1860", act_short="PPC", year=1860,
                  pattern=SECTION_RE, label=lambda n: f"Section {n}"),
    "criminal procedure": dict(act="Code of Criminal Procedure, 1898", act_short="CrPC",
                               year=1898, pattern=SECTION_RE,
                               label=lambda n: f"Section {n}"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chunks.json")
    args = ap.parse_args()

    docs = from_hf_pakistan_laws(list(ACTS.keys()))
    chunks, review = [], 0

    for doc in docs:
        key = next((k for k in ACTS if k in str(doc["file_name"]).lower()), None)
        if not key:
            continue
        spec = ACTS[key]
        for s in split_sections(doc["content"], spec["pattern"]):
            review += s.pop("needs_review")
            chunks.append({
                "id": f"{spec['act_short'].lower()}-{spec['year']}-s{s['section'].lower()}",
                "act": spec["act"], "act_short": spec["act_short"], "year": spec["year"],
                "chapter": "", "section": s["section"],
                "section_label": spec["label"](s["section"]),
                "marginal_note": s["marginal_note"], "text": s["text"],
                "jurisdiction": "federal", "status": "unknown",
                "source_url": "https://pakistancode.gov.pk", "verified": False,
            })

    json.dump({"chunks": chunks}, open(args.out, "w"), indent=1)
    print(f"{len(chunks)} chunks -> {args.out}")
    print(f"{review} contain amendment language and need manual status review")
    print("\nNow read 20 random chunks before you trust any of this:")
    print(f"  python -c \"import json,random;"
          f"[print(c['section_label'],'|',c['marginal_note'],'|',c['text'][:90],'\\n') "
          f"for c in random.sample(json.load(open('{args.out}'))['chunks'],20)]\"")


if __name__ == "__main__":
    main()
