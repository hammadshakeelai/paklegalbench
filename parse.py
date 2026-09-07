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


def load_raw_dataset(cache_path: str = "_local/pdf_data.json") -> list[dict]:
    """Loads AyeshaJadoon/Pakistan_Laws_Dataset: 969 acts as {file_name, text}."""
    import os
    from pathlib import Path
    import urllib.request

    p = Path(cache_path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/datasets/AyeshaJadoon/Pakistan_Laws_Dataset/resolve/main/pdf_data.json"
        print(f"Downloading Pakistan_Laws_Dataset (~46.9MB) to {cache_path}...")
        urllib.request.urlretrieve(url, str(p))
        print("Download complete.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_core_acts(docs: list[dict]) -> dict[str, dict]:
    """Identify Constitution, PPC, and CrPC by preamble/title text rather than random MD5 file names."""
    found = {}
    for doc in docs:
        txt = doc.get("text", "")
        if not txt:
            continue
        cleaned_head = txt[:4000].replace("\xa0", " ")
        cleaned_lower = cleaned_head.lower()

        # Despace single-letter PDF OCR artifact (e.g. 'T H E   C O D E' -> 'THE CODE')
        despaced = re.sub(r"(?<=[a-zA-Z])\s+(?=[a-zA-Z](?:\s|$))", "", cleaned_lower)

        full_clean_text = txt.replace("\xa0", " ")
        if "constitution of the islamic republic of pakistan" in cleaned_lower and len(txt) > 200000:
            found["constitution"] = {"doc": doc, "text": full_clean_text}
        elif "pakistan penal code" in cleaned_lower and len(txt) > 300000:
            found["penal"] = {"doc": doc, "text": full_clean_text}
        elif ("thecodeofcriminalprocedure" in despaced or "code of criminal procedure" in cleaned_lower) and len(txt) > 500000:
            found["criminal procedure"] = {"doc": doc, "text": full_clean_text}
    return found


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
    ap.add_argument("--dataset", default="_local/pdf_data.json")
    args = ap.parse_args()

    raw_docs = load_raw_dataset(args.dataset)
    core_acts = find_core_acts(raw_docs)
    print(f"Matched {len(core_acts)} core acts: {list(core_acts.keys())}")

    chunks, review = [], 0
    for key, spec in ACTS.items():
        if key not in core_acts:
            print(f"Warning: {key} not found in dataset!")
            continue
        act_info = core_acts[key]
        act_text = act_info["text"]
        for s in split_sections(act_text, spec["pattern"]):
            review += s.pop("needs_review")
            chunks.append({
                "id": f"{spec['act_short'].lower()}-{spec['year']}-s{s['section'].lower()}",
                "act": spec["act"], "act_short": spec["act_short"], "year": spec["year"],
                "chapter": "", "section": s["section"],
                "section_label": spec["label"](s["section"]),
                "marginal_note": s["marginal_note"], "text": s["text"],
                "jurisdiction": "federal", "status": "in_force",
                "source_url": "https://pakistancode.gov.pk", "verified": False,
            })

    json.dump({"chunks": chunks}, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"{len(chunks)} chunks -> {args.out}")
    print(f"{review} contain amendment language and need manual status review")
    print("\nNow read 20 random chunks before you trust any of this:")
    print(f"  python -c \"import json,random;"
          f"[print(c['section_label'],'|',c['marginal_note'],'|',c['text'][:90],'\\n') "
          f"for c in random.sample(json.load(open('{args.out}', encoding='utf-8'))['chunks'], min(20, len(chunks)))]\"")


if __name__ == "__main__":
    main()

