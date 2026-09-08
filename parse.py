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

# Matches "302. Punishment of qatl-i-amd.", "4[489F. Dishonestly issuing a cheque.__", and "497. When bail may be taken.\xad(1)"
SECTION_RE = re.compile(
    r"^\s*(?:\[|\d{1,3}\[|\d{1,3}\s+)?(?P<num>\d+[-–]?[A-Z]{0,2})\s*\.\s+(?P<note>[^\n.]{3,120})\.(?:__|--|\s*[\xad–—\-]?\s*\(|\s+)",
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
            "marginal_note": m.group("note").replace("\xad", "-").strip(),
            "text": re.sub(r"\s+", " ", body.replace("\xad", "-")),
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


def parse_constitution(text: str) -> list[dict]:
    """Parse the 1973 Constitution cleanly:
    - Strips the extensive Table of Contents (pages 1-47).
    - Stops before the First Schedule to prevent schedule line numbers colliding with Articles.
    - Merges adjacent marginal note and body lines.
    - Filters out any TOC dotted line artifacts.
    """
    m_start = re.search(r"\bPART\s+I\s+Introductory\b", text, re.I)
    sub = text[m_start.start():] if m_start else text
    m280 = re.search(r"^\s*(?:\d{1,3}\s+)?280\.\s+", sub, re.M)
    if m280:
        m_sched = re.search(r"\bFIRST\s+SCHEDULE\b", sub[m280.start():], re.I)
        if m_sched:
            sub = sub[:m280.start() + m_sched.start()]

    art_pat = re.compile(
        r"^\s*(?:\d{1,3}\s+)?(?P<num>\d+[A-Z]?)\.\s+(?P<note>[A-Z][A-Za-z0-9\s,\-\(\)\'\’\–]{2,100}?)\s*(?:\.|\s*[\r\n])(?!\s*\(|\s*shall|\s*provided|\s*every|\s*no\b)",
        re.M
    )
    raw = split_sections(sub, art_pat)
    merged = []
    i = 0
    while i < len(raw):
        curr = raw[i]
        if i + 1 < len(raw) and raw[i + 1]["section"] == curr["section"]:
            nxt = raw[i + 1]
            body = (nxt["marginal_note"] + ". " + nxt["text"]).strip()
            merged.append({
                "section": curr["section"],
                "marginal_note": curr["marginal_note"].strip(),
                "text": body,
                "needs_review": curr["needs_review"] or nxt["needs_review"]
            })
            i += 2
        else:
            merged.append(curr)
            i += 1

    # Final guard: drop any residual dotted TOC fragments
    return [
        s for s in merged
        if not re.search(r"\.{4,}", s["marginal_note"]) and not re.search(r"\.{4,}", s["text"][:100])
    ]


def clean_crpc_ocr(text: str) -> str:
    """Repair widespread PDF OCR intra-word spacing in Code of Criminal Procedure (1898).
    In official gazette scans, words are separated by 2+ spaces, while individual letters
    or syllables are spaced by single spaces (e.g. 'Exam i n ati on   of   w i tn e s s e s').
    """
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        words = re.split(r"\s{2,}", line.strip())
        fixed_words = []
        for w in words:
            fixed_w = re.sub(r"(?<=[a-zA-Z0-9])\s+(?=[a-zA-Z0-9])", "", w)
            fixed_words.append(fixed_w)
        cleaned_lines.append(" ".join(fixed_words))
    return "\n".join(cleaned_lines)


def make_chunk_id(act_short: str, year: int, section: str) -> str:
    sec_clean = section.lower().replace("-", "").replace("–", "").strip()
    if act_short == "Constitution":
        return f"const-{year}-a{sec_clean}"
    return f"{act_short.lower()}-{year}-s{sec_clean}"


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

        if key == "constitution":
            sections = parse_constitution(act_text)
        elif key == "criminal procedure":
            cleaned_crpc = clean_crpc_ocr(act_text)
            sections = split_sections(cleaned_crpc, spec["pattern"])
            sections = [
                s for s in sections
                if not re.search(r"\.{4,}", s["marginal_note"]) and not re.search(r"\.{4,}", s["text"][:100])
            ]
        else:
            sections = split_sections(act_text, spec["pattern"])
            sections = [
                s for s in sections
                if not re.search(r"\.{4,}", s["marginal_note"]) and not re.search(r"\.{4,}", s["text"][:100])
            ]

        for s in sections:
            if not re.match(r"^\d+[-–]?[A-Z]{0,2}$", s["section"]):
                continue
            review += s.pop("needs_review")
            chunks.append({
                "id": make_chunk_id(spec["act_short"], spec["year"], s["section"]),
                "act": spec["act"], "act_short": spec["act_short"], "year": spec["year"],
                "chapter": "", "section": s["section"],
                "section_label": spec["label"](s["section"]),
                "marginal_note": s["marginal_note"], "text": s["text"],
                "jurisdiction": "federal", "status": "in_force",
                "source_url": "https://pakistancode.gov.pk", "verified": False,
            })

    # Deduplicate provisions by canonical ID, retaining the substantive body (longest text)
    by_id = {}
    for c in chunks:
        cid = c["id"]
        if cid not in by_id or len(c["text"]) > len(by_id[cid]["text"]):
            by_id[cid] = c
    deduped_chunks = list(by_id.values())

    json.dump({"chunks": deduped_chunks}, open(args.out, "w", encoding="utf-8"), indent=1)
    print(f"{len(deduped_chunks)} unique chunks -> {args.out} (from {len(chunks)} raw matches)")
    print(f"{review} contain amendment language and need manual status review")
    print("\nNow read 20 random chunks before you trust any of this:")
    print(f"  python -c \"import json,random;"
          f"[print(c['section_label'],'|',c['marginal_note'],'|',c['text'][:90],'\\n') "
          f"for c in random.sample(json.load(open('{args.out}', encoding='utf-8'))['chunks'], min(20, len(deduped_chunks)))]\"")


if __name__ == "__main__":
    main()

