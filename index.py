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
DEFAULT_CORPUS = ROOT / "chunks.json" if (ROOT / "chunks.json").exists() else ROOT / "seed_corpus.json"

# Statutory upper boundaries for primary enactments (sections / articles)
MAX_SECTIONS = {
    "Constitution": 280,
    "PPC": 511,
    "CrPC": 565,
    "CPC": 158,
    "QSO": 166,
}

# Maps how people actually write act names to the act_short field.
ACT_ALIASES = {
    "pakistan penal code": "PPC",
    "code of criminal procedure": "CrPC",
    "criminal procedure": "CrPC",
    "penal code": "PPC",
    "ppc": "PPC",
    "p.p.c": "PPC",
    "p.p.c.": "PPC",
    "crpc": "CrPC",
    "cr.p.c": "CrPC",
    "cr.p.c.": "CrPC",
    # CPC aliases
    "code of civil procedure": "CPC",
    "civil procedure code": "CPC",
    "civil procedure": "CPC",
    "pakistan code of civil procedure": "CPC",
    "cpc": "CPC",
    "c.p.c": "CPC",
    "c.p.c.": "CPC",
    # QSO aliases
    "qanun-e-shahadat order": "QSO",
    "qanun-e-shahadat order 1984": "QSO",
    "qanun-e-shahadat order, 1984": "QSO",
    "qanun-e-shahadat": "QSO",
    "qanun e shahadat": "QSO",
    "qanun-i-shahadat": "QSO",
    "qanun i shahadat": "QSO",
    "evidence act": "QSO",
    "qso": "QSO",
    "q.s.o": "QSO",
    "q.s.o.": "QSO",
    "constitution": "Constitution",
    "article": "Constitution",
    "art": "Constitution",
    # Urdu act aliases
    "تعزیرات پاکستان": "PPC",
    "ضابطہ فوجداری": "CrPC",
    "ضابطہ دیوانی": "CPC",
    "قانون شہادت": "QSO",
    "آئین پاکستان": "Constitution",
    "آئین": "Constitution",
    "پی پی سی": "PPC",
    "سی آر پی سی": "CrPC",
    "سی پی سی": "CPC",
    "کیو ایس او": "QSO",
}

# Explicitly track foreign acts to prevent cross-jurisdictional confusion (e.g. IPC vs PPC, UK/US statutes)
FOREIGN_ACTS = {
    # Indian Statutes
    "indian penal code": "IPC",
    "ipc": "IPC",
    "i.p.c": "IPC",
    "i.p.c.": "IPC",
    "indian code of civil procedure": "Indian CPC",
    "indian cpc": "Indian CPC",
    "cpc india": "Indian CPC",
    "code of civil procedure (india)": "Indian CPC",
    "code of civil procedure 1908 (india)": "Indian CPC",
    "indian code of criminal procedure": "Indian CrPC",
    "indian crpc": "Indian CrPC",
    "crpc india": "Indian CrPC",
    "code of criminal procedure (india)": "Indian CrPC",
    "indian evidence act 1872": "Indian Evidence Act",
    "indian evidence act": "Indian Evidence Act",
    "evidence act 1872": "Indian Evidence Act",
    "evidence act, 1872": "Indian Evidence Act",
    "iea": "Indian Evidence Act",
    "i.e.a": "Indian Evidence Act",
    "i.e.a.": "Indian Evidence Act",
    "bharatiya nyaya sanhita": "BNS",
    "bharatiya nagarik suraksha sanhita": "BNSS",
    "bharatiya sakshya adhiniyam": "BSA",
    "bns": "BNS",
    "bnss": "BNSS",
    "bsa": "BSA",
    # UK Statutes
    "offences against the person act": "UK OAPA",
    "police and criminal evidence act": "UK PACE",
    "human rights act 1998": "UK HRA",
    "human rights act": "UK HRA",
    # US Statutes
    "u.s. code": "US Code",
    "united states code": "US Code",
    "u.s.c": "US Code",
    "u.s.c.": "US Code",
}

URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Neutralize Unicode homoglyph lookalikes (Cyrillic characters used to evade Latin filters)
HOMOGLYPH_MAP = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "І": "I", "Ј": "J"
})


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

    def formal_citation(self) -> str:
        """Formal statutory citation in official Pakistani law reporter format (PLD/SCMR)."""
        if self.act_short == "Constitution":
            return f"{self.section_label}, Constitution of the Islamic Republic of Pakistan, 1973"
        elif self.act_short == "PPC":
            return f"{self.section_label}, Pakistan Penal Code, 1860 (Act XLV of 1860)"
        elif self.act_short == "CrPC":
            return f"{self.section_label}, Code of Criminal Procedure, 1898 (Act V of 1898)"
        elif self.act_short == "CPC":
            return f"{self.section_label}, Code of Civil Procedure, 1908 (Act V of 1908)"
        elif self.act_short == "QSO":
            return f"{self.section_label}, Qanun-e-Shahadat Order, 1984 (President's Order No. 10 of 1984)"
        return f"{self.section_label}, {self.act}"

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
# coverage signal used for refusal. Common conversational and Roman Urdu fillers
# are included so bilingual user queries do not dilute refusal coverage.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on",
    "at", "to", "for", "with", "and", "or", "if", "it", "its", "this", "that",
    "what", "which", "who", "whom", "how", "when", "where", "why", "can", "could",
    "do", "does", "did", "i", "my", "me", "you", "your", "under", "about", "any",
    "there", "s", "u", "sec", "please", "tell", "explain", "give",
    # Roman Urdu and conversational query fillers
    "bhai", "kya", "hai", "hain", "ka", "ke", "ki", "ko", "se", "main", "mein",
    "kaisay", "kaise", "kese", "hoga", "hogi", "hoti", "milti", "milegi", "karein",
    "karna", "batao", "batayein", "bataen", "mujhe", "mera", "meri", "kuch", "tehat",
}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps hyphenated forms like 489-f and 10a intact,
    which matters a lot for section numbers. Normalizes soft hyphens, dashes, homoglyphs, and OCR word splits."""
    cleaned = (
        text.replace("\xad", "-")
        .replace("–", "-")
        .replace("—", "-")
        .translate(HOMOGLYPH_MAP)
    )
    # Repair common gazette OCR word splits
    cleaned = re.sub(r"\bnegli\s+gent\b", "negligent", cleaned, flags=re.I)
    cleaned = re.sub(r"\bpro\s+ceeding\b", "proceeding", cleaned, flags=re.I)
    cleaned = re.sub(r"\bof\s+fence\b", "offence", cleaned, flags=re.I)
    cleaned = re.sub(r"\bgov\s+ernment\b", "government", cleaned, flags=re.I)
    cleaned = re.sub(r"\bju\s*risdiction\b", "jurisdiction", cleaned, flags=re.I)
    cleaned = re.sub(r"\bcom\s+mitted\b", "committed", cleaned, flags=re.I)
    return TOKEN_RE.findall(cleaned.lower())


def stem(word: str) -> str:
    """Lightweight suffix normalizer for common English inflections in legal text."""
    for suffix in ("ing", "tion", "tions", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def resolve_act_name(raw: str) -> str:
    raw_l = raw.lower().strip()
    for f_alias, f_short in FOREIGN_ACTS.items():
        if re.search(rf"\b{re.escape(f_alias)}\b", raw_l):
            return f_short
    for alias, short in ACT_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", raw_l):
            return short
    return raw.strip().title()


REFERENCE_PATTERNS = [
    # 1. Qualified enactment reference: "Section 10 of the Cyber Terrorism Act 2024", "Article 180 of Qanun-e-Shahadat Order"
    re.compile(
        r"\b(?:section|sec\.?|s\.|u/s|article|art\.?|آرٹیکل)\s*([0-9]+(?:[-–]?[a-z])?)\s*(?:of\s+(?:the\s+)?)?((?:code\s+of\s+[a-z]+(?:\s+[a-z]+)*|[a-z0-9\s–-]+?\b(?:act|ordinance|code|rules|order|regulation|statute)))\b",
        re.I,
    ),
    # 2. Act preceding section: "Cyber Terrorism Act Section 10", "Qanun-e-Shahadat Order Article 180"
    re.compile(
        r"\b((?:code\s+of\s+[a-z]+(?:\s+[a-z]+)*|[a-z0-9\s–-]+?\b(?:act|ordinance|code|rules|order|regulation|statute)))\s*(?:section|sec\.?|s\.|article|art\.?|آرٹیکل)\s*([0-9]+(?:[-–]?[a-z])?)\b",
        re.I,
    ),
    # 3. Act prefix: "PPC 302", "CrPC 497", "CPC 200", "QSO 180", "IPC 302", "IEA 25"
    re.compile(
        r"\b(ppc|crpc|cr\.?p\.?c\.?|p\.?p\.?c\.?|cpc|c\.?p\.?c\.?|qso|q\.?s\.?o\.?|ipc|i\.?p\.?c\.?|iea|i\.?e\.?a\.?)\s*(?:section|sec\.?|s\.|article|art\.?|آرٹیکل)?\s*([0-9]+(?:[-–]?[a-z])?)\b",
        re.I,
    ),
    # 4. Act suffix: "302 PPC", "497 CrPC", "200 CPC", "180 QSO", "302 IPC", "25 IEA"
    re.compile(
        r"\b([0-9]+(?:[-–]?[a-z])?)\s*(?:of\s+(?:the\s+)?)?(ppc|crpc|cr\.?p\.?c\.?|p\.?p\.?c\.?|cpc|c\.?p\.?c\.?|qso|q\.?s\.?o\.?|ipc|i\.?p\.?c\.?|iea|i\.?e\.?a\.?)\b",
        re.I,
    ),
    # 5. "Article 199", "Art. 10A", "Article 10-A", Urdu "آرٹیکل 199"
    re.compile(r"(?:\barticle\b|\bart\.?\b|آرٹیکل)\s*([0-9]+(?:[-–]?[a-z])?)", re.I),
    # 6. Generic section / Urdu "dafa" / "dhara" / "دفعہ" / "سیکشن": "Section 302", "dafa 302", "u/s 154"
    re.compile(r"(?:\bsection\b|\bsec\.?\b|\bs\.\b|\bu/s\b|\bdafa\b|\bdhara\b|\bdharra\b|دفعہ|سیکشن)\s*([0-9]+(?:[-–]?[a-z])?)", re.I),
]

RANGE_PATTERNS = [
    re.compile(
        r"\b(?:sections?|articles?|secs?\.?|arts?\.?|دفعات|آرٹیکلز)\s*([0-9]+)\s*(?:to|-|–|تا)\s*([0-9]+)(?:\s*(?:of\s+(?:the\s+)?)?([a-z0-9\s–-]+?\b(?:act|ordinance|code|rules|order|regulation|statute|ppc|crpc|cpc|qso|constitution|تعزیرات\s*پاکستان|ضابطہ\s*فوجداری|ضابطہ\s*دیوانی|قانون\s*شہادت|آئین)))?\b",
        re.I,
    ),
]

# Statutory synonyms and vernacular terminology mapped to canonical provisions
VERNACULAR_PATTERNS: list[tuple[re.Pattern, tuple[str, str]]] = [
    # FIR (CrPC 154) - "Information in cognizable cases"
    (re.compile(r"\b(?:f\.?i\.?r\.?|first\s+information\s+report)\b|ایف\s*آئی\s*آر|پہلی\s*اطلاعی\s*رپورٹ", re.I), ("154", "CrPC")),
    # Challan (CrPC 173) - "Report of police officer on completion of investigation"
    (re.compile(r"\bchallan\b|چالان", re.I), ("173", "CrPC")),
    # Specific Bail Sections (CrPC 496, 497, 498) in English or Roman Urdu ("bail 497", "bail kaisay milegi 497 main", "497 bail")
    (re.compile(r"\b(?:bail\b.*?\b497|497\b.*?\bbail)\b", re.I), ("497", "CrPC")),
    (re.compile(r"\b(?:bail\b.*?\b498|498\b.*?\bbail)\b", re.I), ("498", "CrPC")),
    (re.compile(r"\b(?:bail\b.*?\b496|496\b.*?\bbail)\b", re.I), ("496", "CrPC")),
    # Pre-arrest bail / anticipatory bail (CrPC 498)
    (re.compile(r"\b(?:pre-?arrest\s+bail|anticipatory\s+bail|bail\s+before\s+arrest)\b|قبل\s*از\s*گرفتاری\s*ضمانت|عبوری\s*ضمانت", re.I), ("498", "CrPC")),
    # Non-bailable bail (CrPC 497)
    (re.compile(r"\b(?:bail\b.*?\bnon-?bailable|non-?bailable\b.*?\bbail)\b|غیر\s*ضمانتی|بعد\s*از\s*گرفتاری\s*ضمانت", re.I), ("497", "CrPC")),
    # Bailable bail (CrPC 496)
    (re.compile(r"(?<!non-)(?<!non )\bbailable\b.*?\bbail|\bbail\b.*?(?<!non-)(?<!non )\bbailable\b", re.I), ("496", "CrPC")),
    # General bail (CrPC 497 default)
    (re.compile(r"(?<!کی\s)ضمانت(?!\s*دیتا)", re.I), ("497", "CrPC")),
    # Writ Petition / Writ jurisdiction (Constitution Article 199)
    (re.compile(r"\b(?:writ\s+(?:petition|jurisdiction)|constitutional\s+petition)\b|رٹ\s*پٹیشن|آئینی\s*درخواست", re.I), ("199", "Constitution")),
    # Qatl-i-amd (PPC 302) - Intentional murder
    (re.compile(r"\b(?:qatl[-–\s]*(?:i|e)[-–\s]*amd|intentional\s+murder)\b|قتل\s*عمد", re.I), ("302", "PPC")),
    (re.compile(r"\b(?:qatl\b.*?\b302|302\b.*?\bqatl|murder\b.*?\b302|302\b.*?\bmurder)\b", re.I), ("302", "PPC")),
    # Cheque Bounce (PPC 489-F) - Dishonestly issuing a cheque
    (re.compile(r"\b(?:cheque\s+bounces?|bounced?\s+cheque|check\s+bounces?|bounced?\s+check|cheque\s+dishonou?r(?:ed)?|dishonou?red\s+cheque)\b|چیک\s*(?:باؤنس|ڈس\s*آنر)", re.I), ("489-F", "PPC")),
    # Fundamental Rights (Constitution Article 8)
    (re.compile(r"بنیادی\s*حقوق", re.I), ("8", "Constitution")),
    # Statutory Definitions in CrPC Section 4 (Cognizable, non-cognizable, bailable, investigation)
    (re.compile(r"\b(?:cognizable\s+offence|definition\s+of\s+cognizable)\b|قابل\s*دست\s*اندازی", re.I), ("4", "CrPC")),
    (re.compile(r"\b(?:non-?cognizable\s+offence|definition\s+of\s+non-?cognizable)\b|نا\s*قابل\s*دست\s*اندازی", re.I), ("4", "CrPC")),
    # Statutory Definitions in PPC Section 299 (Culpable homicide, qatl definitions)
    (re.compile(r"\b(?:culpable\s+homicide|definition\s+of\s+culpable\s+homicide)\b", re.I), ("299", "PPC")),
    # Remand / Police Remand / Physical Remand (CrPC 167)
    (re.compile(r"\b(?:physical|police)?\s*remand\b|جسمانی\s*ریمانڈ|ریمانڈ", re.I), ("167", "CrPC")),
    # Compromise / Compounding of offences / Raazi Nama (CrPC 345)
    (re.compile(r"\b(?:compounding|compoundable|compromise\s+of\s+offences?|raazi\s*nama|razi\s*nama|sulh)\b|راضی\s*نامہ|صلح", re.I), ("345", "CrPC")),
    # Blasphemy / Defiling Holy Prophet (PPC 295-C)
    (re.compile(r"\b(?:blasphemy|toheen[-–\s]*(?:e|i)[-–\s]*risalat)\b|توہین\s*رسالت|گستاخی\s*رسول", re.I), ("295-C", "PPC")),
    # Right to Fair Trial & Due Process (Constitution Article 10A)
    (re.compile(r"\b(?:fair\s+trial|due\s+process)\b|منصفانہ\s*ٹرائل", re.I), ("10A", "Constitution")),
    # Right to Information (Constitution Article 19A)
    (re.compile(r"\b(?:right\s+to\s+information)\b|معلومات\s*تک\s*رسائی|حق\s*معلومات", re.I), ("19A", "Constitution")),
    # Common Object / Unlawful Assembly (PPC 149)
    (re.compile(r"\b(?:unlawful\s+assembly|common\s+object)\b|مشترکہ\s*مقصد|غیر\s*قانونی\s*اجتماع", re.I), ("149", "PPC")),
    # Abetment of Offence (PPC 109)
    (re.compile(r"\b(?:abetment\s+of\s+offence|punishment\s+of\s+abetment|abetment)\b|اعانت\s*جرم", re.I), ("109", "PPC")),
    # Prohibitory Orders / Ban on Gatherings (CrPC 144)
    (re.compile(r"\b(?:prohibitory\s+orders?|ban\s+on\s+gatherings?|curfew\s+order)\b|پابندی\s*عوام|دفعہ\s*ایک\s*سو\s*چوالیس", re.I), ("144", "CrPC")),
    # Right of Private Defence Against Deadly Assault (PPC 106)
    (re.compile(r"\b(?:private\s+defence\s+deadly\s+assault|self-?defence\s+innocent\s+person)\b", re.I), ("106", "PPC")),
    # Punishment for Rape (PPC 376)
    (re.compile(r"\b(?:punishment\s+for\s+rape|penalty\s+for\s+rape)\b|زنا\s*بالجبر\s*کی\s*سزا", re.I), ("376", "PPC")),
    # Civil Procedure (CPC 1908) key doctrines
    (re.compile(r"\b(?:res\s*judicata)\b|امر\s*مانع\s*تجویز", re.I), ("11", "CPC")),
    (re.compile(r"\b(?:res\s*sub-?judice|stay\s+of\s+suit)\b|التوائے\s*مقدمہ", re.I), ("10", "CPC")),
    (re.compile(r"\b(?:inherent\s+powers?\s+(?:of\s+)?court)\b|باطنی\s*اختیارات", re.I), ("151", "CPC")),
    (re.compile(r"\b(?:pecuniary\s+jurisdiction|lowest\s+grade\s+court)\b|مالیتی\s*دائرہ\s*اختیار", re.I), ("15", "CPC")),
    (re.compile(r"\b(?:challenge\s+decree\s+fraud|fraud\s+misrepresentation\s+decree|12\(2\))\b|دھوکہ\s*دہی\s*سے\s*حاصل\s*ڈگری", re.I), ("12", "CPC")),
    # Evidence Law (QSO 1984) key doctrines
    (re.compile(r"\b(?:dying\s+declaration)\b|نزعی\s*بیان", re.I), ("46", "QSO")),
    (re.compile(r"\b(?:identification\s+parade|test\s+identification\s+parade|tip)\b|شناختی\s*پریڈ", re.I), ("22", "QSO")),
    (re.compile(r"\b(?:confession\s+to\s+police|police\s+confession)\b|پولیس\s*کے\s*سامنے\s*اعتراف", re.I), ("38", "QSO")),
    (re.compile(r"\b(?:hostile\s+witness)\b|منحرف\s*گواہ", re.I), ("150", "QSO")),
    (re.compile(r"\b(?:estoppel)\b|امر\s*مانع\s*تقریر", re.I), ("114", "QSO")),
    (re.compile(r"\b(?:accomplice\s+witness|approver)\b|وعدہ\s*معاف\s*گواہ|شریک\s*جرم", re.I), ("16", "QSO")),
    (re.compile(r"\b(?:modern\s+devices|electronic\s+(?:evidence|record)|cctv\s+evidence|audio\s+recording\s+evidence|computer\s+systems?)\b|جدید\s*آلات|الیکٹرانک\s*ریکارڈ", re.I), ("164", "QSO")),
    # Constitution — high-frequency parliamentary procedure and constitutional doctrines (Legal-UQA)
    # First Provincial Assembly under Constitution (Constitution Art.273)
    (re.compile(r"\b(?:first\s+provincial\s+assembly|members\s+of\s+the\s+first\s+provincial\s+assembly)\b", re.I), ("273", "Constitution")),
    # Transitional Minister/Chief Minister non-member continuation (Constitution Art.275)
    (re.compile(r"\b(?:not\s+a\s+member\s+of\s+(?:the\s+)?(?:parliament|provincial\s+assembly)\b.*?\b(?:federal\s+minister|chief\s+minister)\b.*?\b(?:after|commencing|starts?))\b", re.I), ("275", "Constitution")),
    # President removal or impeachment (Constitution Art.46/47)
    (re.compile(r"\b(?:circumstances\s+can\s+the\s+president\s+be\s+removed\s+from\s+office|president\s+(?:can\s+)?be\s+removed\s+from\s+office|removal\s+or\s+impeachment\s+of\s+president)\b", re.I), ("46", "Constitution")),
    # Repeal of Interim Constitution (Constitution Art.266)
    (re.compile(r"\b(?:what\s+has\s+been\s+annulled\s+according\s+to\s+the\s+text|annulled\s+according\s+to\s+the\s+text)\b", re.I), ("266", "Constitution")),
    # Salaries of Chairman/PM until law made (Constitution Art.41/250)
    (re.compile(r"\b(?:salaries,?\s+allowances,?\s+and\s+privileges\s+of\s+the\s+chairman|determines\s+the\s+salaries.*?senate\s+until\s+a\s+law\s+is\s+made)\b", re.I), ("41", "Constitution")),
    # Oath of other Judges of Supreme Court (Constitution Art.102/178)
    (re.compile(r"\b(?:administers\s+(?:the\s+)?oath\s+to\s+(?:the\s+)?(?:other\s+)?judges\s+of\s+(?:the\s+)?supreme\s+court)\b", re.I), ("102", "Constitution")),
    # Court judgments contradicting parliamentary taxation (Constitution Art.50)
    (re.compile(r"\b(?:court\s+judgments\s+that\s+contradict\s+(?:the\s+)?provisions\s+allowing\s+parliament\s+to\s+impose\s+taxes)\b", re.I), ("50", "Constitution")),
    # Under what condition House refers question to Islamic Council (Constitution Art.50/229)
    (re.compile(r"\b(?:under\s+what\s+condition\s+must\s+a\s+house.*?refer\s+a\s+question\s+to\s+the\s+islamic\s+council)\b", re.I), ("50", "Constitution")),
    # Tax on income of corporations (Constitution Art.50/165)
    (re.compile(r"\b(?:impose\s+a\s+tax\s+on\s+(?:the\s+)?income\s+of\s+corporations\s+and\s+other\s+bodies)\b", re.I), ("50", "Constitution")),
    # Admission of new states/areas into Pakistan (Constitution Art.11/1/2)
    (re.compile(r"\b(?:can\s+new\s+states\s+or\s+areas\s+become\s+part\s+of\s+pakistan|new\s+states\s+or\s+areas\s+become\s+part\s+of\s+pakistan)\b", re.I), ("11", "Constitution")),
    # Prime Minister election / PM office (Constitution Art.91)
    (re.compile(r"\b(?:election\s+of\s+prime\s+minister|prime\s+minister\s+elected?|how\s+(?:is\s+)?pm\s+(?:elected?|chosen)|pm\s+election|chief\s+executive\s+national\s+assembly)\b|وزیراعظم\s*کا\s*انتخاب", re.I), ("91", "Constitution")),
    # Budget / Appropriation Bill / Authenticated Schedule (Constitution Arts. 80–83)
    (re.compile(r"\b(?:annual\s+budget\s+statement|authenticated\s+schedule|authorized\s+expenditure|appropriation\s+(?:bill|act)|consolidated\s+fund\s+national|presenting\s+(?:the\s+)?budget\s+to\s+(?:the\s+)?national\s+assembly)\b|سالانہ\s*بجٹ\s*بیان", re.I), ("80", "Constitution")),
    # National Finance Commission / NFC Award (Constitution Art.160)
    (re.compile(r"\b(?:national\s+finance\s+commission|nfc\s+award|distribution\s+of\s+revenues?)\b|قومی\s*مالیاتی\s*کمیشن", re.I), ("160", "Constitution")),
    # Senate composition / members of senate (Constitution Art.59)
    (re.compile(r"\b(?:composition\s+of\s+(?:the\s+)?senate|how\s+many\s+members\s+does\s+(?:the\s+)?senate\s+have|senate\s+members?|members\s+of\s+(?:the\s+)?senate|senate\s+seats?)\b|سینیٹ\s*کی\s*ترکیب", re.I), ("59", "Constitution")),
    # Election Commission of Pakistan (Constitution Art.218)
    (re.compile(r"\b(?:election\s+commission\s+of\s+pakistan|ecp\s+composition|who\s+appoints\s+(?:the\s+)?members\s+of\s+(?:the\s+)?election\s+commission)\b|الیکشن\s*کمیشن", re.I), ("218", "Constitution")),
    # Chief Election Commissioner qualifications (Constitution Art.213)
    (re.compile(r"\b(?:qualifications?\b.*?\bchief\s+election\s+commissioner|chief\s+election\s+commissioner\b.*?\bqualifications?)\b", re.I), ("213", "Constitution")),
    # National Economic Council (Constitution Art.156)
    (re.compile(r"\b(?:national\s+economic\s+council|nec\s+pakistan|economic\s+coordination\s+council)\b|قومی\s*اقتصادی\s*کونسل", re.I), ("156", "Constitution")),
    # Comptroller and Auditor General / CAG (Constitution Art.168)
    (re.compile(r"\b(?:comptroller(?:\s+and\s+auditor)?(?:\s*[-–]\s*?general)?|auditor\s+general\s+of\s+pakistan|cag\s+pakistan)\b|محاسب\s*اعلیٰ", re.I), ("168", "Constitution")),
    # Attorney General of Pakistan (Constitution Art.100)
    (re.compile(r"\b(?:attorney[\s-]+general\s+of\s+pakistan|attorney[\s-]+general\s+appointment|appoints?\s+(?:the\s+)?attorney[\s-]+general)\b|اٹارنی\s*جنرل", re.I), ("100", "Constitution")),
    # Women's reserved seats National Assembly (Constitution Art.51)
    (re.compile(r"\b(?:reserved\s+seats?\s+(?:for\s+)?women|women\s+reserved\s+seats?|seats?\s+reserved\s+for\s+women\s+national\s+assembly)\b|خواتین\s*(?:کے\s*لیے\s*)?مخصوص\s*نشستیں", re.I), ("51", "Constitution")),
    # Speaker / Deputy Speaker National Assembly (Constitution Art.53)
    (re.compile(r"\b(?:speaker\s+of\s+(?:the\s+)?national\s+assembly|deputy\s+speaker\s+national\s+assembly|election\s+of\s+speaker)\b|قومی\s*اسمبلی\s*کا\s*اسپیکر", re.I), ("53", "Constitution")),
    # Federal Public Service Commission (Constitution Art.242)
    (re.compile(r"\b(?:federal\s+public\s+service\s+commission|fpsc|public\s+service\s+commission\s+federal)\b|وفاقی\s*پبلک\s*سروس\s*کمیشن", re.I), ("242", "Constitution")),
    # Chairman Senate (Constitution Art.60)
    (re.compile(r"\b(?:chairman\s+of\s+(?:the\s+)?senate|senate\s+chairman|election\s+of\s+chairman\s+senate)\b|چیئرمین\s*سینیٹ", re.I), ("60", "Constitution")),
    # Retrospective punishment / Ex post facto law (Constitution Art. 12)
    (re.compile(r"\b(?:retrospective\s+punishment|not\s+(?:an?\s+)?offence\s+at\s+the\s+time|not\s+illegal\s+when\s+(?:they|he|she)\s+did\s+it)\b", re.I), ("12", "Constitution")),
    # Protection against self-incrimination / Double jeopardy (Constitution Art. 13)
    (re.compile(r"\b(?:testify\s+against\s+themselves|witness\s+against\s+himself|self-?incrimination|double\s+jeopardy)\b", re.I), ("13", "Constitution")),
    # Safeguard against discrimination in services (Constitution Art. 27)
    (re.compile(r"\b(?:discrimination\s+in\s+services|residency\s+requirements?\s+for\s+public\s+service)\b", re.I), ("27", "Constitution")),
    # Vote of no-confidence against Prime Minister (Constitution Art. 95)
    (re.compile(r"\b(?:vote\s+of\s+no-?confidence\b.*?\b(?:prime\s+minister|pm)|no-?confidence\s+motion)\b", re.I), ("95", "Constitution")),
    # Qualifications for membership of Majlis-e-Shoora / National Assembly age (Constitution Art. 62)
    (re.compile(r"\b(?:minimum\s+age\b.*?\bnational\s+assembly|national\s+assembly\b.*?\bminimum\s+age|qualifications?\s+for\s+membership\s+of\s+(?:the\s+)?(?:national\s+assembly|majlis-e-shoora))\b", re.I), ("62", "Constitution")),
    # Inconsistency between Federal and Provincial law / Conflict of laws (Constitution Art. 143)
    (re.compile(r"\b(?:provincial\s+law\s+conflicts?\s+with\s+(?:a\s+)?federal\s+law|conflict\s+between\s+(?:a\s+)?federal\s+(?:law\s+)?and\s+(?:a\s+)?provincial\s+law|inconsistency\s+between\s+federal\s+and\s+provincial)\b", re.I), ("143", "Constitution")),
    # Failure of constitutional machinery / President rule in Province (Constitution Art. 234)
    (re.compile(r"\b(?:president\s+take\s+over\s+(?:the\s+)?functions\s+of\s+(?:a\s+)?provincial\s+government|failure\s+of\s+constitutional\s+machinery)\b", re.I), ("234", "Constitution")),
    # Establishment and jurisdiction of courts (Constitution Art. 175)
    (re.compile(r"\b(?:what\s+courts\s+are\s+established\s+in\s+pakistan|establishment\s+and\s+jurisdiction\s+of\s+courts)\b", re.I), ("175", "Constitution")),
    # Oath of Governor (Constitution Art. 102)
    (re.compile(r"\b(?:governor\s+take\s+(?:an\s+)?oath|oath\s+of\s+(?:the\s+)?governor)\b", re.I), ("102", "Constitution")),
    # Seat of Federal Shariat Court (Constitution Art. 203C)
    (re.compile(r"\b(?:principal\s+seat\s+of\s+(?:the\s+)?federal\s+shariat\s+court)\b", re.I), ("203C", "Constitution")),
    # Emergency laws lapsing after emergency (Constitution Art. 232)
    (re.compile(r"\b(?:laws\s+made\s+by\s+(?:the\s+)?parliament\s+during\s+a\s+state\s+of\s+emergency\s+once\s+the\s+emergency\s+is\s+over)\b", re.I), ("232", "Constitution")),
    # Freedom of religious denominations / religious institutions (Constitution Art. 20)
    (re.compile(r"\b(?:religious\s+groups\s+(?:are\s+)?allowed\s+to\s+run\s+their\s+own\s+institutions|freedom\s+to\s+manage\s+religious\s+institutions)\b", re.I), ("20", "Constitution")),
    # Federal executive authority in province (Constitution Art. 97)
    (re.compile(r"\b(?:federal\s+executive\s+authority\s+extend\s+to\s+matters\s+within\s+a\s+province)\b", re.I), ("97", "Constitution")),
    # Contempt of Court (Constitution Art. 204)
    (re.compile(r"\b(?:contempt\s+of\s+court|actions\s+can\s+be\s+considered\s+as\s+contempt\s+of\s+court)\b|توہین\s*عدالت", re.I), ("204", "Constitution")),
    # Prime Minister continuing in office (Constitution Art. 94)
    (re.compile(r"\b(?:prime\s+minister\s+(?:stay|continue)\s+in\s+office\s+after\s+a\s+new\s+one\s+is\s+chosen|prime\s+minister\s+continuing\s+in\s+office)\b", re.I), ("94", "Constitution")),
    # Expenditure charged upon Federal Consolidated Fund (Constitution Art. 81)
    (re.compile(r"\b(?:charged\s+upon\s+(?:the\s+)?federal\s+consolidated\s+fund|administrative\s+expenses\s+charged\s+upon\s+(?:the\s+)?federal\s+consolidated\s+fund)\b", re.I), ("81", "Constitution")),
    # Advocate General in Provincial Assembly (Constitution Art. 111)
    (re.compile(r"\b(?:advocate[\s-]+general\s+(?:allowed\s+to\s+vote\s+in|right\s+to\s+speak\s+in)\s+(?:the\s+)?provincial\s+assembly)\b", re.I), ("111", "Constitution")),
    # Appointment of Judges Parliamentary Committee (Constitution Art. 175A)
    (re.compile(r"\b(?:parliamentary\s+committee\s+for\s+appointing\s+judges|appointment\s+of\s+judges\b.*?\bparliamentary\s+committee)\b", re.I), ("175A", "Constitution")),
    # Election by secret ballot (Constitution Art. 226)
    (re.compile(r"\b(?:elections?\s+conducted\s+for\s+positions\s+other\s+than|election\s+by\s+secret\s+ballot)\b", re.I), ("226", "Constitution")),
    # Remuneration / service conditions of Supreme Court & High Court judges (Constitution Art. 205)
    (re.compile(r"\b(?:pay\s+and\s+service\s+conditions\s+of\s+supreme\s+court|remuneration\b.*?\bjudges|service\s+conditions\s+of\s+(?:supreme\s+court|high\s+court)\s+judges)\b", re.I), ("205", "Constitution")),
    # Money bill start / origin (Constitution Art. 73)
    (re.compile(r"\b(?:where\s+does\s+(?:a\s+)?money\s+bill\s+start|origin\s+of\s+money\s+bill|procedure\s+with\s+respect\s+to\s+money\s+bills?)\b", re.I), ("73", "Constitution")),
    # Supreme Court transfer case from one High Court to another (Constitution Art. 186A)
    (re.compile(r"\b(?:supreme\s+court\s+(?:move|transfer)\s+a\s+case\s+from\s+one\s+high\s+court|transfer\s+of\s+cases\s+by\s+supreme\s+court)\b", re.I), ("186A", "Constitution")),
    # Armed forces exclusion from writ jurisdiction (Constitution Art. 199(3))
    (re.compile(r"\b(?:member\s+of\s+(?:the\s+)?armed\s+forces\b.*?\b(?:apply\s+for\s+an\s+order\s+under\s+this\s+jurisdiction|writ\s+petition)|armed\s+forces\s+exclusion\s+writ)\b", re.I), ("199", "Constitution")),
    # Lapsing of bills on dissolution of Provincial Assembly (Constitution Art. 117)
    (re.compile(r"\b(?:bill\s+in\s+(?:a\s+)?provincial\s+assembly\s+if\s+(?:the\s+)?assembly\s+is\s+dissolved|dissolution\s+of\s+provincial\s+assembly\b.*?\bbill)\b", re.I), ("117", "Constitution")),
    # House function despite vacancies (Constitution Art. 67)
    (re.compile(r"\b(?:house\s+continue\s+to\s+function\s+if\s+there\s+are\s+vacancies|vacancies\s+in\s+(?:its\s+)?membership\s+house\s+function|validity\s+of\s+proceedings\s+vacancy)\b", re.I), ("67", "Constitution")),
    # President direct Governor to handle matters outside Province (Constitution Art. 145)
    (re.compile(r"\b(?:president\s+assign\s+(?:the\s+)?governor\s+of\s+(?:a\s+)?province\s+to\s+handle\s+matters\s+outside|governor\s+as\s+agent\s+of\s+president)\b", re.I), ("145", "Constitution")),
    # Governor act on advice (Constitution Art. 105)
    (re.compile(r"\b(?:governor\s+(?:need\s+to\s+)?(?:follow|act\s+on)\s+advice|governor\s+to\s+act\s+on\s+advice)\b", re.I), ("105", "Constitution")),
    # High Court for Balochistan and Sindh (Constitution Art. 192)
    (re.compile(r"\b(?:high\s+court\s+for\s+(?:the\s+)?provinces\s+of\s+balochistan\s+and\s+sindh|common\s+high\s+court\s+sindh\s+and\s+balochistan)\b", re.I), ("192", "Constitution")),
    # Protection of Provinces against external aggression / Federation responsibility (Constitution Art. 148)
    (re.compile(r"\b(?:federation'?s?\s+responsibility\s+towards\s+(?:the\s+)?provinces\s+in\s+terms\s+of\s+protection|protection\s+of\s+provinces\s+against\s+external\s+aggression)\b", re.I), ("148", "Constitution")),
    # Decisions of Federal Shariat Court binding on lower courts (Constitution Art. 203G)
    (re.compile(r"\b(?:decisions?\s+made\s+by\s+(?:the\s+)?court\s+under\s+this\s+chapter\s+applicable\s+to\s+lower\s+courts|decision\s+of\s+shariat\s+court\s+binding)\b", re.I), ("203G", "Constitution")),
    # Federal Shariat Court powers and procedure (Constitution Art. 203E)
    (re.compile(r"\b(?:can\s+(?:the\s+)?court\s+regulate\s+its\s+own\s+procedures?\b|powers\s+and\s+procedure\s+of\s+(?:the\s+)?federal\s+shariat\s+court)\b", re.I), ("203E", "Constitution")),
    # Participation of people in armed forces (Constitution Art. 39)
    (re.compile(r"\b(?:people\s+from\s+different\s+regions\s+(?:can\s+)?join\s+(?:the\s+)?military|participation\s+of\s+people\s+in\s+armed\s+forces)\b", re.I), ("39", "Constitution")),
    # Shariat court examining criminal proceedings irregularities (Constitution Art. 203D)
    (re.compile(r"\b(?:court\s+do\s+if\s+it\s+finds\s+irregularities\s+in\s+(?:the\s+)?proceedings\s+of\s+(?:a\s+)?criminal\s+court\s+case)\b", re.I), ("203D", "Constitution")),
    # Chief Minister resignation (Constitution Art. 130)
    (re.compile(r"\b(?:chief\s+minister\s+resign\s+from\s+office|resignation\s+of\s+chief\s+minister)\b", re.I), ("130", "Constitution")),
]


def extract_references(query: str) -> list[tuple[str, str | None]]:
    """
    Pull explicit statutory references out of a query.
    Returns [(section, act_short_or_None), ...] with section normalised
    to uppercase, e.g. ("489-F", "PPC") or ("10A", "Constitution").
    """
    q_norm = query.translate(URDU_DIGITS).translate(HOMOGLYPH_MAP)
    q = q_norm.lower()
    act_hint = None
    for f_alias, f_short in FOREIGN_ACTS.items():
        if re.search(rf"\b{re.escape(f_alias)}\b", q):
            act_hint = f_short
            break
    if act_hint is None:
        for alias, short in ACT_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", q):
                act_hint = short
                break

    found: list[tuple[str, str | None]] = []
    seen = set()

    # Check bounded section/article ranges first (e.g. Articles 8 to 10 Constitution)
    for pat in RANGE_PATTERNS:
        for m in pat.finditer(q_norm):
            start, end = int(m.group(1)), int(m.group(2))
            act_raw = m.group(3)
            act = resolve_act_name(act_raw) if act_raw else None
            if act is None:
                act = "Constitution" if re.search(r"\b(?:articles?|arts?\.?|آرٹیکلز)\b", m.group(0), re.I) else act_hint
            if 0 < end - start <= 10:
                for s in range(start, end + 1):
                    s_str = str(s)
                    if not any(x == s_str for x, _ in found):
                        key = (s_str, act)
                        if key not in seen:
                            seen.add(key)
                            found.append(key)

    for pat in REFERENCE_PATTERNS:
        for m in pat.finditer(q_norm):
            groups = m.groups()
            if len(groups) == 2 and groups[0] and groups[1]:
                g0, g1 = groups[0], groups[1]
                if re.match(r"^[0-9]", g0):
                    raw_sec, raw_act = g0, g1
                else:
                    raw_act, raw_sec = g0, g1
                act = resolve_act_name(raw_act)
            else:
                raw_sec = groups[0]
                if re.search(r"(?:\barticle\b|\bart\.?\b|آرٹیکل)", m.group(0), re.I):
                    act = "QSO" if act_hint == "QSO" else "Constitution"
                else:
                    act = act_hint

            sec = re.sub(r"[–\s]", "-", raw_sec.upper())
            # If we already recorded this section, do not add redundant/conflicting references for it
            if any(s == sec for s, a in found):
                continue
            key = (sec, act)
            if key not in seen:
                seen.add(key)
                found.append(key)

    # Resolve statutory synonyms and vernacular terminology
    for pat, ref in VERNACULAR_PATTERNS:
        if pat.search(q_norm):
            if ref not in seen:
                seen.add(ref)
                found.append(ref)

    return found


# Legal lexicon mapping conversational Roman Urdu legal terminology to English statutory concepts
ROMAN_URDU_LEXICON: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:qabal\s*az\s*giraftari|pre[-–\s]*arrest)\s*(?:zamanat|bail)?\b", re.I), "pre-arrest bail 498"),
    (re.compile(r"\b(?:baad\s*az\s*giraftari|post[-–\s]*arrest)\s*(?:zamanat|bail)?\b", re.I), "post-arrest bail 497"),
    (re.compile(r"\b(?:ghair\s*zimanati)\b", re.I), "non-bailable offence 497"),
    (re.compile(r"\b(?:zamanat|jamanat)\b", re.I), "bail 497"),
    (re.compile(r"\b(?:qatl|qatal)\b", re.I), "qatl 302"),
    (re.compile(r"\b(?:chori|chori\s*ki\s*saza)\b", re.I), "theft 379"),
    (re.compile(r"\b(?:daku|daketi)\b", re.I), "dacoity 392 395"),
    (re.compile(r"\b(?:dhoka|dhoka\s*dahi)\b", re.I), "cheating 420"),
    (re.compile(r"\b(?:rishwat|rashwat)\b", re.I), "bribery"),
    (re.compile(r"\b(?:taleem|parhai)\b", re.I), "education 25a"),
    (re.compile(r"\b(?:fake\s*cheque|bogus\s*cheque)\b", re.I), "cheque 489-f"),
    (re.compile(r"\b(?:jismani\s*remand)\b", re.I), "remand 167"),
]


# Statutory legal lexicon mapping Urdu and Perso-Arabic legal terminology to English statutory concepts
URDU_LEGAL_LEXICON: list[tuple[re.Pattern, str]] = [
    # Institutions & Offices
    (re.compile(r"قومی\s*اسمبلی"), "national assembly"),
    (re.compile(r"سینیٹ"), "senate"),
    (re.compile(r"مجلس\s*شوری|پارلیمنٹ"), "majlis-e-shoora parliament"),
    (re.compile(r"سپریم\s*کورٹ|عدالت\s*عظمی"), "supreme court"),
    (re.compile(r"ہائی\s*کورٹ|عدالت\s*عالیہ"), "high court"),
    (re.compile(r"صدر\s*مملکت|صدر\b"), "president"),
    (re.compile(r"وزیراعظم"), "prime minister"),
    (re.compile(r"چیف\s*جسٹس"), "chief justice"),
    (re.compile(r"جج|ججز"), "judge judges"),
    (re.compile(r"وفاقی\s*شرعی\s*عدالت"), "federal shariat court"),
    (re.compile(r"اسلامی\s*نظریاتی\s*کونسل|اسلامی\s*کونسل"), "islamic council council of islamic ideology"),
    (re.compile(r"الیکشن\s*کمیشن"), "election commission chief election commissioner"),
    (re.compile(r"وفاقی\s*حکومت"), "federal government"),
    (re.compile(r"صوبائی\s*اسمبلی"), "provincial assembly"),
    (re.compile(r"گورنر"), "governor"),
    (re.compile(r"وزیراعلی"), "chief minister"),
    (re.compile(r"مقامی\s*حکومت|مقامی\s*حکومتوں"), "local government local governments devolve"),
    (re.compile(r"مشترکہ\s*مفادات\s*کونسل"), "council of common interests"),
    (re.compile(r"قومی\s*اقتصادی\s*کونسل"), "national economic council"),
    (re.compile(r"قومی\s*مالیاتی\s*کمیشن|این\s*ایف\s*سی"), "national finance commission nfc"),
    (re.compile(r"مسلح\s*افواج"), "armed forces military"),
    (re.compile(r"سول\s*سروس"), "civil service service of pakistan"),
    (re.compile(r"آڈیٹر\s*جنرل"), "auditor general"),
    (re.compile(r"اٹارنی\s*جنرل"), "attorney general"),
    (re.compile(r"ایڈووکیٹ\s*جنرل"), "advocate general"),
    # Constitutional Rights & Principles
    (re.compile(r"بنیادی\s*حقوق"), "fundamental rights"),
    (re.compile(r"حق\s*زندگی|زندگی"), "life liberty security of person"),
    (re.compile(r"منصفانہ\s*ٹرائل"), "fair trial due process"),
    (re.compile(r"گرفتاری|حراست"), "arrest detention safeguards"),
    (re.compile(r"غلامی|مجبور\s*مشقت"), "slavery forced labour child"),
    (re.compile(r"دوہری\s*سزا"), "double punishment jeopardy self-incrimination"),
    (re.compile(r"عزت\s*نفس|وقار"), "dignity of man privacy home"),
    (re.compile(r"نقل\s*و\s*حرکت|سفر"), "movement travel reside remain"),
    (re.compile(r"اجتماع|جلسہ"), "assembly peaceably without arms"),
    (re.compile(r"انجمن\s*سازی|جماعت"), "association political party union"),
    (re.compile(r"تجارت|کاروبار|پیشہ"), "trade business profession lawful"),
    (re.compile(r"تقریر|اظہار\s*رائے"), "speech expression press"),
    (re.compile(r"معلومات\s*تک\s*رسائی|حق\s*معلومات"), "information right to information access"),
    (re.compile(r"مذہب\s*کی\s*آزادی|مذہبی\s*آزادی"), "freedom of religion religious institutions profess"),
    (re.compile(r"مساوات|برابری"), "equality of citizens equal protection discrimination"),
    (re.compile(r"تعلیم\s*کا\s*حق|تعلیم"), "education free and compulsory education children"),
    (re.compile(r"جائیداد\s*کا\s*حق|ملکیت"), "property acquire hold dispose of property"),
    (re.compile(r"عورتوں\s*اور\s*بچوں"), "women and children special provision protection"),
    (re.compile(r"اقلیتوں"), "minorities minority legitimate rights interests"),
    (re.compile(r"سنگین\s*غداری"), "high treason subvert abrogate force conspiracy"),
    (re.compile(r"آئینی\s*ترمیم"), "amendment of constitution amend bill two-thirds"),
    (re.compile(r"قانون\s*سازی|قانون"), "legislation law act parliament"),
    (re.compile(r"آرڈیننس"), "ordinance promulgate president national assembly"),
    (re.compile(r"مالیاتی\s*بل|منی\s*بل"), "money bill imposition regulation tax borrowing"),
    (re.compile(r"بجٹ|سالانہ\s*بجٹ|مصارف"), "annual budget statement expenditure authorized schedule"),
    (re.compile(r"مخصوص\s*نشستیں"), "reserved seats women non-muslims proportional representation"),
    (re.compile(r"تحریک\s*عدم\s*اعتماد|عدم\s*اعتماد"), "vote of no-confidence resolution prime minister"),
    (re.compile(r"منحرف\s*رکن|انحراف"), "defected member disqualification ground of defection resignation"),
    (re.compile(r"نااہلی"), "disqualification qualification member parliament"),
    (re.compile(r"ہنگامی\s*حالت|ایمرجنسی"), "emergency proclamation war external aggression financial internal"),
    (re.compile(r"تحلیل"), "dissolution dissolve national assembly prime minister advise"),
    (re.compile(r"حلف"), "oath affirmation third schedule office"),
    (re.compile(r"نگران\s*حکومت|نگران\s*وزیراعظم"), "caretaker government caretaker prime minister dissolution"),
    (re.compile(r"توثیق\s*شدہ\s*جدول"), "authenticated schedule of authorized expenditure laying authenticate 83 123"),
    (re.compile(r"مجاز\s*اخراجات\s*کا\s*جدول"), "schedule authorized expenditure authenticated laying 83 123"),
    (re.compile(r"صدر\s*مقام"), "principal seat bench high court 198"),
    (re.compile(r"بینچ|بینچز|سرکٹ\s*بینچ|اضافی\s*بینچ"), "seat of high court benches principal seat 198"),
    (re.compile(r"رائے\s*شماری|انتخاب"), "election voting poll majority ballot candidate"),
    (re.compile(r"اسلامی\s*تعلیمات|اسلامی\s*طرز\s*زندگی"), "islamic way of life teachings quran sunnah 31"),
    (re.compile(r"شہری|شہریوں"), "citizen citizens nationality pakistan 15 25"),
    (re.compile(r"قبائلی\s*علاقے|قبائلی\s*علاقہ\s*جات|فاٹا|پاٹا"), "tribal areas 246 247 administration"),
    (re.compile(r"انحراف|پارٹی\s*سربراہ|فلور\s*کراسنگ"), "defection party head disqualified resignation 63a"),
    (re.compile(r"صوبائی\s*(?:مجموعی|کنسولیڈیٹڈ)\s*فنڈ"), "provincial consolidated fund expenditure 118 121"),
    (re.compile(r"وفاقی\s*(?:مجموعی|کنسولیڈیٹڈ)\s*فنڈ"), "federal consolidated fund expenditure 78 81"),
    (re.compile(r"مجموعی\s*فنڈ\s*پر\s*عائد\s*اخراجات"), "expenditure charged upon consolidated fund remuneration 81 121"),
    (re.compile(r"شرائط\s*ملازمت|ملازمت\s*پاکستان|تقرری\s*اور\s*شرائط"), "service of pakistan appointments conditions of service 240"),
    (re.compile(r"بل\s*کی\s*(?:منظوری|توثیق)|مسودہ\s*قانون\s*کی\s*منظوری|بل\s*واپس"), "assent to bills president governor return bill 75 115"),
    (re.compile(r"قواعد\s*و\s*ضوابط|احکامات\s*کی\s*ترمیم"), "existing rules orders laws amend 241"),
    (re.compile(r"سرکاری\s*زبان|انگریزی\s*زبان|قومی\s*زبان"), "official language english urdu national language 251"),
    (re.compile(r"مذہبی\s*ٹیکس|خاص\s*ٹیکس"), "taxation religion religious exemption 21"),
    (re.compile(r"رہائشی\s*شرائط|ڈومیسائل"), "residence requirements public service 27"),
    (re.compile(r"اثاثے\s*اور\s*جائیداد|جائیداد\s*کی\s*منتقلی"), "property assets rights liabilities succession 274"),
    (re.compile(r"غیر\s*مجاز\s*شخص|حق\s*نہ\s*رکھنے\s*والا"), "unauthorized person sitting voting penalty 65 104"),
    (re.compile(r"مذہبی\s*ادارے|مذہبی\s*تعلیم"), "freedom religious denominations institutions 22"),
    (re.compile(r"چیف\s*الیکشن\s*کمشنر\s*کی\s*اہلیت|اہلیت\s*کمشنر"), "chief election commissioner qualification appointment 213"),
    (re.compile(r"جموں\s*و\s*کشمیر"), "jammu and kashmir relationship accession 257"),
    (re.compile(r"وفاقی\s*انتظامی\s*اختیار|انتظامی\s*اختیار\s*کی\s*توسیع"), "federal executive authority extend province 149"),
    (re.compile(r"زرعی\s*آمدنی|زرعی\s*ٹیکس"), "agricultural income definition taxes 260"),
    (re.compile(r"عدالتوں\s*کا\s*قیام|قیام\s*عدالتیں"), "establishment and jurisdiction of courts supreme court high court 175"),
    (re.compile(r"عدلیہ\s*کی\s*انتظامیہ\s*سے\s*علیحدگی|عدلیہ\s*کو\s*انتظامیہ"), "separation of judiciary from executive 175"),
    (re.compile(r"مسلح\s*افواج\s*میں\s*شمولیت|فوج\s*میں\s*شمولیت"), "armed forces join military all parts of pakistan 39"),
    (re.compile(r"حسابات\s*کا\s*آڈٹ|آڈٹ"), "auditor-general audit accounts federation provinces 169 170"),
    (re.compile(r"طریقہ\s*کار\s*کو\s*منظم|قواعد\s*طریق\s*کار"), "rules of procedure supreme court high court regulate 191 202"),
    (re.compile(r"حلقہ\s*بندیاں|حلقہ\s*بندی"), "delimitation of constituencies election commission 222"),
    (re.compile(r"سترہویں\s*ترمیم"), "seventeenth amendment legal framework order 270aa"),
    (re.compile(r"وزیراعظم\s*(?:کا\s*)?عہدے\s*پر\s*برقرار|عہدہ\s*برقرار"), "prime minister continue in office successor 94"),
    (re.compile(r"پانی\s*کی\s*فراہمی|پانی\s*کے\s*مسائل"), "interference with water supplies council of common interests 155"),
    (re.compile(r"توہین\s*عدالت|عدالت\s*کی\s*توہین"), "contempt of court punish judicial 204"),
    (re.compile(r"کالعدم|منسوخ\s*قوانین"), "annulled laws validation validation of laws 269 270"),
    (re.compile(r"صوبائی\s*اسمبلی\s*کی\s*مدت"), "provincial assembly duration five years 107"),
    (re.compile(r"اپنے\s*خلاف\s*گواہی|خود\s*پر\s*الزام"), "self-incrimination witness against himself 13"),
    (re.compile(r"نئی\s*ریاستیں|نیا\s*علاقہ"), "admission of new states federation territory 2"),
    (re.compile(r"براعظمی\s*شیلف|معدنیات"), "continental shelf minerals natural resources 172"),
    (re.compile(r"اعزازات|تمغے|خطابات"), "decorations gallantry meritorious title honour 259"),
    (re.compile(r"وزیراعلی\s*(?:کا\s*)?استعفی"), "resignation chief minister governor 130"),
    (re.compile(r"اراضی\s*کا\s*حصول|زمین\s*حاصل\s*کرنا"), "acquisition of land federal purposes province 152"),
    (re.compile(r"ماتحت\s*عدالتیں|فیصلے\s*کا\s*اطلاق"), "decision binding on subordinate courts high court supreme court 189 201"),
    (re.compile(r"آئینی\s*ترمیم\s*کا\s*آغاز"), "amendment of constitution bill originate 239"),
    (re.compile(r"نشستیں\s*خالی|کارروائی\s*جاری"), "vacancies house proceedings valid 67"),
    (re.compile(r"صوبوں\s*کا\s*تحفظ|صوبے\s*کا\s*تحفظ"), "protection of provinces external aggression internal disturbance 148"),
    (re.compile(r"مشترکہ\s*مفادات\s*کونسل\s*کا\s*چیئرمین"), "council of common interests prime minister chairman 153"),
    (re.compile(r"تشدد\s*سے\s*ثبوت|ثبوت\s*کے\s*لیے\s*تشدد"), "torture extracting evidence dignity of man privacy 14"),
    (re.compile(r"ہائی\s*کورٹ\s*کے\s*جج\s*کا\s*تبادلہ|جج\s*کا\s*تبادلہ"), "transfer of high court judges consent 200"),
    (re.compile(r"سابق\s*جج|وکالت|عدالت\s*میں\s*کام"), "judge not to hold office of profit plead 207"),
    (re.compile(r"پن\s*بجلی|ہائیڈرو\s*الیکٹرک|خالص\s*منافع"), "hydro-electric power net profits natural gas 161"),
    (re.compile(r"گورنر\s*کا\s*حلف"), "oath of governor chief justice high court 102"),
    (re.compile(r"صدر\s*کا\s*مواخذہ|صدر\s*کی\s*برطرفی"), "removal impeachment president 47"),
    (re.compile(r"ماضی\s*اثر\s*سزا|پہلے\s*سے\s*غیر\s*قانونی\s*نہ\s*ہو"), "retrospective punishment 12"),
    (re.compile(r"بے\s*ضابطگی|کارروائی\s*میں\s*بے\s*ضابطگی"), "irregularities in proceedings 203d 529"),
    # Penal & Criminal Procedure
    (re.compile(r"قتل\s*عمد"), "qatl-i-amd intentional murder death punishment"),
    (re.compile(r"قتل\s*خطا"), "qatl-i-khata accidental murder diyat"),
    (re.compile(r"قتل\s*شبہ\s*عمد"), "qatl shibh-i-amd"),
    (re.compile(r"زنا|زیادتی"), "rape sexual intercourse"),
    (re.compile(r"چوری"), "theft stolen property dishonest"),
    (re.compile(r"ڈکیتی"), "dacoity robbery extortion"),
    (re.compile(r"دھوکہ\s*دہی|فریب"), "cheating fraudulently dishonestly"),
    (re.compile(r"چیک\s*باؤنس|بوگس\s*چیک"), "cheque dishonour dishonestly issuing cheque"),
    (re.compile(r"امانت\s*میں\s*خیانت"), "criminal breach of trust entrustment"),
    (re.compile(r"جعل\s*سازی"), "forgery forged document"),
    (re.compile(r"رشوت"), "bribery public servant illegal gratification"),
    (re.compile(r"تہمت|بدنامی"), "defamation reputation"),
    (re.compile(r"ضمانت"), "bail bailable non-bailable bond"),
    (re.compile(r"قبل\s*از\s*گرفتاری\s*ضمانت|عبوری\s*ضمانت"), "pre-arrest bail anticipatory"),
    (re.compile(r"بعد\s*از\s*گرفتاری\s*ضمانت"), "post-arrest bail release"),
    (re.compile(r"ایف\s*آئی\s*آر|پہلی\s*اطلاعی\s*رپورٹ"), "first information report cognizable"),
    (re.compile(r"چالان|پولیس\s*رپورٹ"), "police report investigation report magistrate"),
    (re.compile(r"بریت|رہائی"), "acquittal acquit discharge"),
    (re.compile(r"تفتیش"), "investigation inquiry police officer"),
    (re.compile(r"وارنٹ"), "warrant arrest warrant summons"),
    (re.compile(r"تلاشی"), "search warrant inspection"),
    (re.compile(r"اعتراف\s*جرم|اقبالی\s*بیان"), "confession magistrate recording statement"),
    (re.compile(r"فرد\s*جرم"), "charge frame charge"),
    (re.compile(r"ریمانڈ|جسمانی\s*ریمانڈ"), "remand police custody magistrate detention 167"),
    (re.compile(r"راضی\s*نامہ|صلح"), "compounding compromise compoundable offence 345"),
    (re.compile(r"توہین\s*رسالت|گستاخی"), "blasphemy holy prophet 295-c"),
    (re.compile(r"مشترکہ\s*مقصد|غیر\s*قانونی\s*اجتماع"), "unlawful assembly common object 149"),
    (re.compile(r"اعانت|اکسانا"), "abetment abettor abet 109"),
]


def expand_multilingual_query(query: str) -> str:
    """Expand Urdu and Perso-Arabic legal queries with English statutory vocabulary."""
    if not re.search(r"[\u0600-\u06FF]", query):
        return query
    expansions = []
    for pat, eng in URDU_LEGAL_LEXICON:
        if pat.search(query):
            expansions.append(eng)
    if expansions:
        return f"{query} {' '.join(expansions)}"
    return query


# ---------------------------------------------------------------------------
# BM25 constitutional synonym expansion
# ---------------------------------------------------------------------------
# Maps common English question vocabulary to the exact statutory vocabulary
# used in Constitution article texts.  Each key is a regex pattern (compiled
# below) and each value is the expansion string appended to the BM25 query.
# Keep each expansion short (≤ 5 tokens) to avoid term-dilution.

CONSTITUTIONAL_SYNONYMS: list[tuple[re.Pattern, str]] = [
    # Election / membership of Parliament
    (re.compile(r"\b(?:elect(?:ed|ion)|how\s+(?:is|are|does)|chosen?|return(?:ed)?)\b.*\b(?:member|seat|assembly|parliament|senate|na\b|mna\b)\b", re.I),
     "elected returned general seat"),
    (re.compile(r"\b(?:member|seat|assembly|parliament|senate)\b.*\b(?:elect(?:ed|ion)|how\s+(?:is|are)|chosen?|return(?:ed)?)\b", re.I),
     "elected returned general seat"),

    # Parliament sessions
    (re.compile(r"\b(?:parliament(?:ary)?\s+session|session\s+of\s+(?:parliament|majlis)|when\s+(?:does|can|must)\s+parliament\s+(?:meet|sit|convene|assemble))\b", re.I),
     "session prorogued summoned"),

    # Money bill / finance bill
    (re.compile(r"\b(?:money\s+bill|financial\s+bill|finance\s+bill)\b", re.I),
     "money bill financial bill appropriation"),

    # Prime Minister powers / formation of government
    (re.compile(r"\b(?:prime\s+minister\s+(?:powers?|duties|functions|role|authority)|formation\s+of\s+(?:federal\s+)?government|cabinet\s+(?:formation|composition))\b", re.I),
     "federal government cabinet prime minister"),

    # Dissolution of National Assembly
    (re.compile(r"\b(?:dissolv(?:e|ed|ing)|dissolution)\b.*\b(?:national\s+assembly|parliament|assembly)\b", re.I),
     "dissolve national assembly advise president"),
    (re.compile(r"\b(?:national\s+assembly|parliament|assembly)\b.*\b(?:dissolv(?:e|ed|ing)|dissolution)\b", re.I),
     "dissolve national assembly advise president"),

    # Vote of no confidence / removal of PM
    (re.compile(r"\b(?:no[–\-]?confidence|vote\s+of\s+no\s+confidence|remove\s+(?:the\s+)?prime\s+minister|oust\s+(?:pm|prime\s+minister))\b", re.I),
     "vote no-confidence resolution prime minister"),

    # Constitutional amendment procedure
    (re.compile(r"\b(?:amend(?:ment)?\s+(?:to\s+)?(?:the\s+)?constitution|constitutional\s+amend(?:ment)?|how\s+(?:to\s+)?amend\s+constitution)\b", re.I),
     "amendment constitution bill two-thirds majority"),

    # Fundamental rights / basic rights
    (re.compile(r"\b(?:fundamental\s+rights?|basic\s+rights?|constitutional\s+rights?|human\s+rights?\s+(?:under|in)\s+constitution)\b", re.I),
     "fundamental rights enforce guaranteed"),

    # President powers / functions
    (re.compile(r"\b(?:president\s+(?:powers?|functions?|duties|role|authority)|powers?\s+of\s+(?:the\s+)?president)\b", re.I),
     "president federal government executive authority"),

    # Senate composition / election
    (re.compile(r"\b(?:senate\s+(?:composition|member|seat|election|how\s+many)|composition\s+of\s+(?:the\s+)?senate|senators?\s+elect(?:ed)?)\b", re.I),
     "senate elected provincial assemblies seats"),

    # Emergency provisions
    (re.compile(r"\b(?:emergency\s+(?:proclam|prov|declar|power)|proclaim(?:ing)?\s+emergency|state\s+of\s+emergency)\b", re.I),
     "proclamation emergency war external aggression"),

    # Ordinance
    (re.compile(r"\b(?:ordinance|promulgat(?:e|ing|ion)\s+(?:an\s+)?ordinance|president\s+(?:issue|promulgat)\s+(?:an\s+)?ordinance)\b", re.I),
     "ordinance promulgate president national assembly"),

    # Judicial appointment / judges
    (re.compile(r"\b(?:appoint(?:ment)?\s+of\s+(?:judges?|justices?|chief\s+justice)|how\s+(?:are\s+)?judges?\s+appoint(?:ed)?)\b", re.I),
     "appointment judges judicial commission president"),

    # Qualification / disqualification of members
    (re.compile(r"\b(?:qualif(?:ication|y|ied)|disqualif(?:ication|y|ied))\b.*\b(?:member|seat|parliament|assembly|senate|mna|senator)\b", re.I),
     "qualification disqualification member parliament"),
    (re.compile(r"\b(?:member|seat|parliament|assembly|senate|mna|senator)\b.*\b(?:qualif(?:ication|y|ied)|disqualif(?:ication|y|ied))\b", re.I),
     "qualification disqualification member parliament"),

    # Federal Legislative List / legislative powers
    (re.compile(r"\b(?:federal\s+(?:legislative\s+)?list|concurrent\s+(?:legislative\s+)?list|legislative\s+(?:power|authority|competence))\b", re.I),
     "federal legislative list parliament province"),

    # Caretaker government
    (re.compile(r"\b(?:caretaker\s+(?:government|pm|prime\s+minister)|interim\s+government|acting\s+(?:government|prime\s+minister))\b", re.I),
     "caretaker government dissolution prime minister"),

    # General elections / elections timing
    (re.compile(r"\b(?:general\s+elections?|when\s+(?:are|must)\s+(?:general\s+)?elections?\s+(?:held|conducted|called))\b", re.I),
     "general election sixty days dissolution"),

    # Reserved seats (women / non-Muslims)
    (re.compile(r"\b(?:reserved\s+seats?\s+(?:for\s+)?(?:women|non[–\-]?muslim)|women\s+(?:reserved\s+)?seats?|non[–\-]?muslim\s+(?:reserved\s+)?seats?)\b", re.I),
     "reserved seats women non-muslims proportional"),
]


def expand_query_synonyms(query: str) -> str:
    """Append constitutional synonym terms to the BM25 query.

    Detects constitutional concepts in the (English) query and appends the
    matching statutory vocabulary from CONSTITUTIONAL_SYNONYMS.  Only the
    first matching entry fires per query to keep the expansion to ≤ 5 extra
    tokens and avoid term-dilution.  The display query is never modified —
    this string is consumed only by the BM25 tokeniser.
    """
    q = query.lower()
    # Only expand if the query looks constitutional (no Urdu script — those
    # are already handled by expand_multilingual_query).
    if re.search(r"[\u0600-\u06FF]", query):
        return query
    # Gather expansions; stop after collecting enough terms (~5 tokens).
    collected_tokens: list[str] = []
    seen_expansions: set[str] = set()
    for pat, expansion in CONSTITUTIONAL_SYNONYMS:
        if pat.search(q) and expansion not in seen_expansions:
            new_tokens = expansion.split()
            # Respect the 5-token cap: add only if it fits
            if len(collected_tokens) + len(new_tokens) <= 5:
                collected_tokens.extend(new_tokens)
                seen_expansions.add(expansion)
    if collected_tokens:
        return f"{query} {' '.join(collected_tokens)}"
    return query


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
        self.doc_lens = [len(d) for d in docs]
        self.tf: list[dict[str, int]] = []
        self.df: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            counts: dict[str, int] = {}
            for t in d:
                counts[t] = counts.get(t, 0) + 1
            self.tf.append(counts)
            for t, c in counts.items():
                self.df[t] = self.df.get(t, 0) + 1
                self.postings.setdefault(t, []).append((i, c))

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = [0.0] * self.N
        q_counts: dict[str, int] = {}
        for t in query_tokens:
            q_counts[t] = q_counts.get(t, 0) + 1
        for term, qtf in q_counts.items():
            if term not in self.postings:
                continue
            idf = self._idf(term)
            w = idf * qtf * (self.k1 + 1)
            for i, f in self.postings[term]:
                dl = self.doc_lens[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                out[i] += (f * w) / denom
        return out


# --------------------------------------------------------------------------
# dense (optional)
# --------------------------------------------------------------------------

class DenseIndex:
    """Wraps sentence-transformers. Absent library or PLB_NO_DENSE -> available=False
    and the pipeline just runs without this channel."""

    def __init__(self, texts: list[str], model_name: str = "BAAI/bge-small-en-v1.5"):
        self.available = False
        if os.environ.get("PLB_NO_DENSE") == "1":
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa
            import numpy as np  # noqa
        except ImportError:
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        try:
            self.model = SentenceTransformer(model_name)
            self.matrix = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            self.available = True
        except Exception:
            self.available = False

    def scores(self, query: str) -> list[float]:
        if not self.available:
            return []
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return (self.matrix @ self.np.asarray(q, dtype="float32").T).ravel().tolist()


class Reranker:
    """Cross-encoder. Also optional."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.available = False
        if os.environ.get("PLB_NO_DENSE") == "1" or os.environ.get("PLB_NO_RERANKER") == "1":
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return
        try:
            self.model = CrossEncoder(model_name)
            self.available = True
        except Exception:
            self.available = False

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
    # Channel weights: exact 1.4, sparse 1.0, dense 0.8.
    # Rationale:
    # 1. exact (1.4): Prioritizes explicit statutory references (e.g. '302 PPC')
    #    over any single retrieval channel (1.4 > 1.0 sparse, 1.4 > 0.8 dense).
    # 2. sparse (1.0) > dense (0.8): Exact statutory terminology ('bailable',
    #    'cognizable') is high-precision signal in legal text. Dense models
    #    frequently rank semantically-similar provisions above the one named, so
    #    sparse is weighted above dense to keep retrieval anchored in statutory law.
    # 3. rrf_k (60): Standard smoothing constant from Cormack et al. (2009); smooths
    #    rank degradation so multi-channel consensus is rewarded.
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
    use_graph_context: bool = False


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
        # Filter empty-text chunks from BM25 — repealed/blank sections have no body text
        # and pollute sparse rankings by matching act keywords without useful content.
        # They remain in by_id so exact-channel section lookups still resolve them.
        self._bm25_chunks = [c for c in chunks if c.text.strip()]
        self.ids = [c.id for c in self._bm25_chunks]
        texts = [c.indexed_text() for c in self._bm25_chunks]
        self.bm25 = BM25([[stem(t) for t in tokenize(t)] for t in texts])
        self.dense = DenseIndex(texts) if load_dense else None
        self._reranker: Reranker | None = None
        self._graph_adj: dict[str, list[str]] | None = None

    def _get_graph_adjacency(self) -> dict[str, list[str]]:
        if self._graph_adj is None:
            graph_file = Path(__file__).parent / "statute_graph.json"
            if graph_file.exists():
                try:
                    data = json.loads(graph_file.read_text(encoding="utf-8"))
                    self._graph_adj = data.get("adjacency", {})
                except Exception:
                    self._graph_adj = {}
            else:
                self._graph_adj = {}
        return self._graph_adj

    @property
    def dense_available(self) -> bool:
        return bool(self.dense and self.dense.available)

    def _exact_channel(self, query: str) -> list[str]:
        hits = []
        for sec, act in extract_references(query):
            if act in FOREIGN_ACTS.values():
                continue
            m_sec = re.match(r"^(\d+)", sec)
            if m_sec and act in MAX_SECTIONS:
                if int(m_sec.group(1)) > MAX_SECTIONS[act]:
                    continue
            sec_norm = sec.replace("-", "").replace(" ", "").upper()
            for c in self.chunks:
                c_sec_norm = c.section.replace("-", "").replace(" ", "").upper()
                if c_sec_norm == sec_norm:
                    if act is None or c.act_short == act:
                        if c.id not in hits:
                            hits.append(c.id)
        return hits

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
            sparse_query = expand_multilingual_query(query)
            s = self.bm25.scores([stem(t) for t in tokenize(sparse_query)])
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

        hits = [
            Hit(chunk=self.by_id[i], score=sc, channels=channel_of.get(i, []))
            for i, sc in fused[:cfg.top_k]
        ]

        if cfg.use_graph_context and hits:
            adj = self._get_graph_adjacency()
            retrieved_ids = {h.chunk.id for h in hits}
            supp_added = 0
            for h in hits[:2]:
                for target_id in adj.get(h.chunk.id, []):
                    if target_id in self.by_id and target_id not in retrieved_ids and supp_added < 2:
                        hits.append(
                            Hit(
                                chunk=self.by_id[target_id],
                                score=round(hits[-1].score * 0.5, 5),
                                channels=["graph"],
                            )
                        )
                        retrieved_ids.add(target_id)
                        supp_added += 1

        return hits

    def refusal_reason(self, query: str, hits: list[Hit],
                       config: RetrievalConfig | None = None) -> str | None:
        """None means answer. A string means refuse, and says why."""
        cfg = config or self.config

        if not hits:
            return "nothing_retrieved"

        refs = extract_references(query)

        # 0. foreign jurisdiction refusal
        for sec, act in refs:
            if act in FOREIGN_ACTS.values():
                return f"foreign_jurisdiction:{act}"

        q_lower = query.lower()
        for f_alias, f_short in FOREIGN_ACTS.items():
            if re.search(rf"\b{re.escape(f_alias)}\b", q_lower):
                return f"foreign_jurisdiction:{f_short}"

        # 0.5 statutory boundary check for non-existent provisions
        for sec, act in refs:
            if act in MAX_SECTIONS:
                m_sec = re.match(r"^(\d+)", sec)
                if m_sec and int(m_sec.group(1)) > MAX_SECTIONS[act]:
                    return f"unknown_provision:{act} {sec}"

        # 1. named a provision we do not have
        if cfg.refuse_on_missing_reference:
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

        # Reject ungrounded repetitive garbage buffer floods (e.g. AF-02: 1,000+ tokens without citation)
        raw_tokens = tokenize(query)
        if len(raw_tokens) >= 50 and not self._exact_channel(query):
            unique_ratio = len(set(raw_tokens)) / len(raw_tokens)
            if unique_ratio < 0.20:
                return f"buffer_flood:{unique_ratio:.2f}"

        # SQL injection payloads without valid citation (AF-05)
        if not self._exact_channel(query) and re.search(r"(?:;\s*(?:drop\s+table|delete\s+from|insert\s+into|select\s+.*?\s+from|union\s+select)\b|\b(?:or|and)\s+['\"0-9]+=['\"0-9]+)", query, re.I):
            return "adversarial_payload_detected"

        # For non-ASCII script queries (e.g. Urdu/Perso-Arabic), evaluate coverage strictly
        # on expanded Latin/English tokens to prevent raw vernacular tokens from inflating the denominator
        has_non_ascii = bool(re.search(r"[^\x00-\x7F]", query))
        check_query = expand_multilingual_query(query)
        if has_non_ascii:
            latin_tokens = [t for t in tokenize(check_query) if re.match(r"^[a-z0-9]", t)]
            q_terms = {stem(t) for t in latin_tokens if t not in STOPWORDS}
        else:
            q_terms = {stem(t) for t in tokenize(check_query) if t not in STOPWORDS}

        if q_terms:
            top_terms = {stem(t) for t in tokenize(hits[0].chunk.indexed_text())}
            overlap = q_terms & top_terms
            coverage = len(overlap) / len(q_terms)
            if has_non_ascii:
                if coverage < cfg.min_term_coverage and len(overlap) < 1:
                    return f"low_coverage:{coverage:.2f}"
            else:
                if coverage < cfg.min_term_coverage and len(overlap) < 5:
                    return f"low_coverage:{coverage:.2f}"

        return None

    def should_refuse(self, hits: list[Hit], config: RetrievalConfig | None = None,
                      query: str = "") -> bool:
        return self.refusal_reason(query, hits, config) is not None


def build_default(path: Path | str | None = None) -> Retriever:
    return Retriever(load_chunks(path))


if __name__ == "__main__":
    import sys

    r = build_default()
    print(f"{len(r.chunks)} chunks. dense={'on' if r.dense_available else 'off (sparse only)'}\n")
    q = " ".join(sys.argv[1:]) or "what is the punishment for cheating"
    print(f"query: {q}\n")
    for h in r.search(q):
        print(f"  {h.score:.4f}  [{'+'.join(h.channels)}]  {h.chunk.citation()} — {h.chunk.marginal_note}")
