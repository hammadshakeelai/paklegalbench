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
    "constitution": "Constitution",
    "article": "Constitution",
    "art": "Constitution",
    # Urdu act aliases
    "تعزیرات پاکستان": "PPC",
    "ضابطہ فوجداری": "CrPC",
    "آئین پاکستان": "Constitution",
    "آئین": "Constitution",
    "پی پی سی": "PPC",
    "سی آر پی سی": "CrPC",
}

# Explicitly track foreign acts to prevent cross-jurisdictional confusion (e.g. IPC vs PPC)
FOREIGN_ACTS = {
    "indian penal code": "IPC",
    "ipc": "IPC",
    "i.p.c": "IPC",
    "i.p.c.": "IPC",
    "indian evidence act": "Indian Evidence Act",
    "bnss": "BNSS",
    "bns": "BNS",
    "bsa": "BSA",
}

URDU_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


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
    which matters a lot for section numbers. Normalizes soft hyphens and en/em dashes."""
    cleaned = text.replace("\xad", "-").replace("–", "-").replace("—", "-")
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
    # 1. Qualified enactment reference: "Section 10 of the Cyber Terrorism Act 2024"
    re.compile(
        r"\b(?:section|sec\.?|s\.|u/s)\s*([0-9]+(?:[-–]?[a-z])?)\s*(?:of\s+(?:the\s+)?)?([a-z0-9\s–-]+?\b(?:act|ordinance|code|rules|order|regulation|statute))\b",
        re.I,
    ),
    # 2. Act preceding section: "Cyber Terrorism Act Section 10"
    re.compile(
        r"\b([a-z0-9\s–-]+?\b(?:act|ordinance|code|rules|order|regulation|statute))\s*(?:section|sec\.?|s\.)\s*([0-9]+(?:[-–]?[a-z])?)\b",
        re.I,
    ),
    # 3. Act prefix: "PPC 302", "CrPC 497", "PPC 303-A", "IPC 302"
    re.compile(
        r"\b(ppc|crpc|cr\.?p\.?c\.?|p\.?p\.?c\.?|ipc|i\.?p\.?c\.?)\s*(?:section|sec\.?|s\.)?\s*([0-9]+(?:[-–]?[a-z])?)\b",
        re.I,
    ),
    # 4. Act suffix: "302 PPC", "497 CrPC", "489-F P.P.C.", "302 IPC"
    re.compile(
        r"\b([0-9]+(?:[-–]?[a-z])?)\s*(?:of\s+the\s+)?(ppc|crpc|cr\.?p\.?c\.?|p\.?p\.?c\.?|ipc|i\.?p\.?c\.?)\b",
        re.I,
    ),
    # 5. "Article 199", "Art. 10A", "Article 10-A", Urdu "آرٹیکل 199"
    re.compile(r"(?:\barticle\b|\bart\.?\b|آرٹیکل)\s*([0-9]+(?:[-–]?[a-z])?)", re.I),
    # 6. Generic section / Urdu "dafa" / "dhara" / "دفعہ" / "سیکشن": "Section 302", "dafa 302", "u/s 154"
    re.compile(r"(?:\bsection\b|\bsec\.?\b|\bs\.\b|\bu/s\b|\bdafa\b|\bdhara\b|\bdharra\b|دفعہ|سیکشن)\s*([0-9]+(?:[-–]?[a-z])?)", re.I),
]

RANGE_PATTERNS = [
    re.compile(
        r"\b(?:sections?|articles?|secs?\.?|arts?\.?|دفعات|آرٹیکلز)\s*([0-9]+)\s*(?:to|-|–|تا)\s*([0-9]+)(?:\s*(?:of\s+(?:the\s+)?)?([a-z0-9\s–-]+?\b(?:act|ordinance|code|rules|order|regulation|statute|ppc|crpc|constitution|تعزیرات\s*پاکستان|ضابطہ\s*فوجداری|آئین)))?\b",
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
    (re.compile(r"ضمانت", re.I), ("497", "CrPC")),
    # Writ Petition / Writ jurisdiction (Constitution Article 199)
    (re.compile(r"\b(?:writ\s+(?:petition|jurisdiction)|constitutional\s+petition)\b|رٹ\s*پٹیشن|آئینی\s*درخواست", re.I), ("199", "Constitution")),
    # Qatl-i-amd (PPC 302) - Intentional murder
    (re.compile(r"\b(?:qatl[-–\s]*(?:i|e)[-–\s]*amd|intentional\s+murder)\b|قتل\s*عمد", re.I), ("302", "PPC")),
    # Cheque Bounce (PPC 489-F) - Dishonestly issuing a cheque
    (re.compile(r"\b(?:cheque\s+bounces?|bounced?\s+cheque|check\s+bounces?|bounced?\s+check|cheque\s+dishonou?r(?:ed)?|dishonou?red\s+cheque)\b|چیک\s*(?:باؤنس|ڈس\s*آنر)", re.I), ("489-F", "PPC")),
    # Fundamental Rights (Constitution Article 8)
    (re.compile(r"بنیادی\s*حقوق", re.I), ("8", "Constitution")),
    # Statutory Definitions in CrPC Section 4 (Cognizable, non-cognizable, bailable, investigation)
    (re.compile(r"\b(?:cognizable\s+offence|definition\s+of\s+cognizable)\b|قابل\s*دست\s*اندازی", re.I), ("4", "CrPC")),
    (re.compile(r"\b(?:non-?cognizable\s+offence|definition\s+of\s+non-?cognizable)\b|نا\s*قابل\s*دست\s*اندازی", re.I), ("4", "CrPC")),
    # Statutory Definitions in PPC Section 299 (Culpable homicide, qatl definitions)
    (re.compile(r"\b(?:culpable\s+homicide|definition\s+of\s+culpable\s+homicide)\b", re.I), ("299", "PPC")),
]


def extract_references(query: str) -> list[tuple[str, str | None]]:
    """
    Pull explicit statutory references out of a query.
    Returns [(section, act_short_or_None), ...] with section normalised
    to uppercase, e.g. ("489-F", "PPC") or ("10A", "Constitution").
    """
    q_norm = query.translate(URDU_DIGITS)
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
                act = "Constitution" if re.search(r"(?:\barticle\b|\bart\.?\b|آرٹیکل)", m.group(0), re.I) else act_hint

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
    (re.compile(r"توثیق\s*شدہ\s*جدول"), "authenticated schedule of authorized expenditure laying authenticate"),
    (re.compile(r"صدر\s*مقام"), "principal seat bench high court"),
    (re.compile(r"رائے\s*شماری|انتخاب"), "election voting poll majority ballot candidate"),
    (re.compile(r"اسلامی\s*تعلیمات|اسلامی\s*طرز\s*زندگی"), "islamic way of life teachings quran sunnah"),
    (re.compile(r"شہری|شہریوں"), "citizen citizens nationality pakistan"),
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
        self.ids = [c.id for c in chunks]
        texts = [c.indexed_text() for c in chunks]
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

        check_query = expand_multilingual_query(query)
        q_terms = {stem(t) for t in tokenize(check_query) if t not in STOPWORDS}
        if q_terms:
            top_terms = {stem(t) for t in tokenize(hits[0].chunk.indexed_text())}
            coverage = len(q_terms & top_terms) / len(q_terms)
            if coverage < cfg.min_term_coverage:
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
