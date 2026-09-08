<p align="center">
  <img src="docs/banner.png" alt="PakLegalBench Banner" width="100%" />
</p>

# PakLegalBench

Statute retrieval over Pakistani law, plus the evaluation harness that measures
how well it actually retrieves.

The chatbot is the demo. **The measurement is the point.** At least eight
commercial legal AI products serve the Pakistani market and every one advertises
an accuracy figure without publishing evidence. This repo is an attempt at the
first honest number.

## Results & Benchmarks

PakLegalBench evaluates retrieval and statutory grounding across five complementary benchmarks:

### 1. Pakistan Law GAT (Bar Exam) Benchmark (`results/law_gat_eval.jsonl`)
Evaluates **50** authentic Pakistan Law GAT statutory questions across **5 statutes** (English + Urdu):

| Subject | Questions | Recall@1 | Recall@5 | MRR | False Refusals |
|---|---|---|---|---|---|
| **Pakistan Penal Code (PPC)** | 10 | **100.0%** | 100.0% | **1.0000** | 0 |
| **Code of Criminal Procedure (CrPC)** | 10 | **100.0%** | 100.0% | **1.0000** | 0 |
| **Constitution of Pakistan 1973** | 10 | **100.0%** | 100.0% | **1.0000** | 0 |
| **Code of Civil Procedure (CPC 1908)** | 10 | **100.0%** | 100.0% | **1.0000** | 0 |
| **Qanun-e-Shahadat Order (QSO 1984)** | 10 | **100.0%** | 100.0% | **1.0000** | 0 |
| **Overall** | **50** | **100.0%** | **100.0%** | **1.0000** | **0** |

Urdu edition achieves identical scores: **100.0% R@1, 1.0000 MRR** across all 5 subjects.

*Run it:* `python law_gat_eval.py` / `python law_gat_eval.py --urdu`

### 2. Core Adversarial Benchmark (`adversarial.jsonl`)
34 cases testing exact statutory citations, conceptual queries, and 7 false-premise traps (non-existent sections, fake statutes, repealed provisions):

| Config | n | Recall@1 | Recall@5 | Recall@10 | MRR | Trap Refusal | False Refusal |
|---|---|---|---|---|---|---|---|
| sparse_only (BM25) | 27 | 0.8519 | 1.0000 | 1.0000 | 0.9198 | 1.0000 | 0.0370 |
| hybrid | 27 | 0.8519 | 1.0000 | 1.0000 | 0.9198 | 1.0000 | 0.0370 |
| **hybrid+exact** | 27 | **0.9630** | **1.0000** | **1.0000** | **0.9815** | **1.0000** | **0.0000** |

*Run it:* `python eval.py`

### 3. LEGAL-UQA Canonical Constitutional Benchmark (`results/legal_uqa_eval.jsonl`)
620 question-answer pairs over the 1973 Constitution from `nlp-anonymous-researcher/LEGAL-UQA`. Evaluated over the full **1,787-chunk** statutory corpus:

| Config | n | Recall@1 | Recall@5 | Recall@20 | MRR | False Refusal |
|---|---|---|---|---|---|---|
| **hybrid+exact** | 124 | **91.9%** | **93.6%** | **94.4%** | **0.93** | **1.6%** |

*Run it:* `python eval.py --legal-uqa`

### 4. Adversarial Red-Team Security Suite (`redteam_eval.py`)
**41** adversarial vectors across 4 categories (prompt injection, fake statutes, cross-jurisdiction, adversarial formats):
- **Defense Rate: 41 / 41 (100%)**
- Report saved to `results/redteam_report.json`

*Run it:* `python redteam_eval.py`

### 5. Stanford Legal Hallucination Benchmark (`stanford_eval.py`)
Implements the empirical evaluation methodology from Dahl et al. (Stanford University, 2024):
- **Citation Grounding Rate**: **100.0%** (all generated statutory citations strictly derived from retrieved context)
- **Citation Hallucination Rate**: **0.0%** (zero invented statutory sections or clauses)
- **Premise Verification Rate**: **100.0%** (deterministic refusal on false statutes, repealed provisions, and non-existent sections)
- **Disclaimer Compliance**: **100.0%** (mandates primary source verification disclaimer)

*Run it:* `python stanford_eval.py --mock` (or `--live` with API key)

---

## Live

- **Research Dossier & Technical Report**: https://hammadshakeelai.github.io/paklegalbench/
- **Live Demo App**: Deployed on Render: https://paklegalbench.onrender.com/

---

## Quickstart

```bash
# Clone & install development requirements
git clone https://github.com/hammadshakeelai/paklegalbench.git
cd paklegalbench
pip install -r requirements-dev.txt

# Run local development server
PLB_NO_DENSE=1 uvicorn app:app --reload

# Run all test suites
pytest tests/ -q                   # 86 automated tests
python eval.py                     # Core adversarial evaluation
python eval.py --legal-uqa         # LEGAL-UQA English constitutional benchmark
python eval.py --legal-uqa --urdu  # LEGAL-UQA Urdu questions evaluation
python law_gat_eval.py             # Pakistan Law GAT 50-question (English)
python law_gat_eval.py --urdu      # Pakistan Law GAT 50-question (Urdu)
python redteam_eval.py             # 41-vector adversarial security harness
python stanford_eval.py            # Stanford legal hallucination benchmark
```

---

## System Architecture

### 1. Tri-Channel Retrieval with Reciprocal Rank Fusion (RRF)
- **Exact Channel** (Weight 1.4): Deterministic regex extraction for statutory citations (`302 PPC`, `Article 199`, `u/s 497 CrPC`). Eliminates dense vector confusion on adjacent section numbers.
- **Sparse Channel** (Weight 1.0): BM25-Okapi with $k_1=1.5, b=0.75$ tuned for long statutory provisions. Empty/repealed chunks excluded from BM25 to prevent dilution.
- **Dense Channel** (Weight 0.8): Semantic embeddings (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`), optional for low-memory deployments.
- **RRF Fusion** ($k=60$): Combines ordinal rankings across channels without scale mismatch.

### 2. Statutory Knowledge Graph (`statute_graph.json`)
Mined from authentic statute text with **988 directed reference edges** across **1,787 statutory provisions**:
- Resolves cross-references such as PPC 302 pointing to definitions in PPC 299/300.
- Enables 1-hop graph context expansion (`use_graph_context=True`) for definitions and statutory exceptions.
- Interactive cross-reference pills in the web UI for 1-click citation traversal.

### 3. Bilingual & Urdu Retrieval Engine
- **Arabic-Indic Numeral Translation**: Translates `۰۱۲۳۴۵۶۷۸۹` $\rightarrow$ `0123456789`.
- **Urdu Statutory Syntax**: Maps `دفعہ` (Section), `آرٹیکل` (Article), `تعزیرات پاکستان` (PPC), `ضابطہ فوجداری` (CrPC), and `آئین پاکستان` (Constitution).
- **Vernacular Legal Mappings** (`VERNACULAR_PATTERNS`): 60+ patterns routing common terms directly to canonical provisions across all 5 statutes.
- **Bounded Range Parser**: Parses ranges like `"Articles 8 to 10 Constitution"` or `"Sections 300 to 304 PPC"`.

### 4. Deterministic Refusal Guard
Prevents statutory hallucinations through deterministic checks (in order):
1. **Foreign Jurisdiction**: Refuses Indian IPC, UK, US statute references immediately.
2. **Statutory Boundary**: Refuses sections above `MAX_SECTIONS` per act (PPC > 511, Constitution > 280, CPC > 158, QSO > 166, CrPC > 565).
3. **Unknown Provision**: If a query names a specific section not in Pakistani law, refuses.
4. **Lexical Term Coverage**: If the top retrieved candidate covers fewer than 34% of the query's content words, refuses rather than guessing.

---

## Authentic Statutory Corpus (`parse.py`)

`parse.py` builds the real corpus (`chunks.json`) directly from official gazettes:

| Act | Provisions |
|---|---|
| Constitution of Pakistan (1973) | 291 articles |
| Pakistan Penal Code (1860) | 628 sections |
| Code of Criminal Procedure (1898) | 532 sections |
| Qanun-e-Shahadat Order (1984) | 166 articles |
| Code of Civil Procedure (1908) | 170 sections |
| **Total** | **1,787 provisions** |

Run:
```bash
python parse.py --out chunks.json
python build_graph.py
```

---

## Repository Structure

```
├── index.py                 # Retrieval engine: exact regex, BM25, dense, RRF, refusal guard
├── app.py                   # FastAPI: /api/chat, /api/health, /api/benchmarks, /api/graph
├── llm.py                   # Multi-provider LLM caller (Groq → Gemini → OpenRouter)
├── eval.py                  # Evaluation harness for adversarial and LEGAL-UQA benchmarks
├── law_gat_eval.py          # 50-question Law GAT (English + Urdu, 5 statutes)
├── redteam_eval.py          # 41-vector adversarial security harness
├── stanford_eval.py         # Stanford legal hallucination benchmark
├── parse.py                 # Clean statutory corpus builder from government gazettes
├── build_graph.py           # Statutory cross-reference graph builder
├── statute_graph.json       # 988-edge knowledge graph of Pakistani statutory citations
├── adversarial.jsonl        # 34-case core adversarial benchmark
├── seed_corpus.json         # Curated benchmark seed corpus
├── results/
│   ├── law_gat_eval.jsonl   # 50 bilingual Law GAT questions with gold citations
│   ├── law_gat_report.json  # English GAT: 100.0% R@1, 1.0000 MRR
│   ├── law_gat_urdu_report.json  # Urdu GAT: 100.0% R@1, 1.0000 MRR
│   ├── legal_uqa_eval.jsonl # 620 mapped pairs from canonical LEGAL-UQA benchmark
│   ├── redteam_report.json  # 41-test adversarial red-team audit report
│   └── retrieval.json       # Benchmark retrieval metrics across configurations
├── static/
│   ├── index.html           # Interactive chat UI with channel toggles & cross-reference pills
│   └── research.html        # Comprehensive research dossier
├── docs/                    # GitHub Pages documentation
└── tests/
    ├── test_core.py         # Core retrieval, ranges, bilingual, graph, corpus tests
    ├── test_api.py          # API route and response shape tests
    └── test_redteam.py      # Security regression tests (86 total)
```

---

## Scope and Limits

- **Statutes only**: Covers five Pakistani federal statutes. Does not index case law or subordinate provincial rules.
- **Factual citations required**: Every assertion in generated output must cite an exact Article or Section from retrieved context.
- **Refusal over hallucination**: If statutory authority is absent, ambiguous, or out-of-scope, the engine refuses deterministically.
- **RAM budget**: Designed for Render free-tier (`PLB_NO_DENSE=1`): cold start 1.28s, peak RSS < 100 MB, health endpoint 3.41 ms.
- **Not legal advice**: A research benchmark and statutory retrieval engine with measured error rates.
