# PakLegalBench Quality Assurance & Verification Report

**Date:** September 8, 2026  
**Auditor:** Antigravity QA Agent  
**Environment:** Python 3.12 (Windows / Render Free Tier Simulation)  
**Corpus:** Pakistani Statutory Law (Constitution 1973, PPC 1860, CrPC 1898)

---

## Executive Summary

A comprehensive quality assurance and reliability audit was performed on the PakLegalBench codebase covering test expansion, CI evaluation gates, Render free-tier resource utilization, and API contracts.

| Category | Target / Requirement | Result | Status |
|---|---|---|---|
| **Unit Test Coverage** | Edge cases: empty, whitespace, symbols, multi-citations | 46/46 tests passing (0.71s runtime) | **PASS** |
| **CI Gate: Recall@1** | $\ge 0.90$ on `hybrid+exact` | **0.9630** (+0.0630 margin) | **PASS** |
| **CI Gate: Trap Refusal Rate** | $\ge 0.85$ on `hybrid+exact` | **1.0000** (100% trap rejection) | **PASS** |
| **CI Gate: False Refusal Rate** | $\le 0.15$ on `hybrid+exact` | **0.0000** (0 false refusals) | **PASS** |
| **Render RAM Budget** | $< 150$ MB RAM under load | **66.77 MB** peak RSS (83.23 MB headroom) | **PASS** |
| **Cold Boot Time** | Fast cold boot | **1.58s** import, **3.27 ms** per-request latency | **PASS** |
| **API Contracts & JSON Schemas** | Strict validation for `/api/chat`, `/api/health`, static endpoints | 18/18 API test cases passing | **PASS** |

---

## 1. Unit Test Expansion (`tests/test_core.py`)

Unit tests were significantly expanded from 11 original tests to 28 core tests (plus 18 API tests, totaling 46 test cases) in `tests/test_core.py` and `tests/test_api.py`.

### Edge Cases Covered:
1. **Empty Queries:**
   - `tokenize("") == []`
   - `extract_references("") == []`
   - `Retriever.search("") == []`
   - `Retriever.refusal_reason("", []) == "nothing_retrieved"`
   - `Retriever.should_refuse([], query="") is True`

2. **Whitespace Queries:**
   - Multi-space, tabs, carriage returns, and newlines (`" \t \n \r "`)
   - Guaranteed empty tokenization and graceful refusal reason (`"nothing_retrieved"`).

3. **Special Symbols & Punctuation:**
   - Pure punctuation strings (`"!@#$%^&*()_+=-~`{}[]|:;'<>,.?/"`) correctly produce zero tokens and trigger deterministic refusal.
   - Punctuation-wrapped statutory citations (`"*** 302 PPC ***"`, `"(u/s 497 Cr.P.C.)"`, `"[Article 199]"`) correctly extract clean statutory identifiers `("302", "PPC")`, `("497", "CrPC")`, and `("199", "Constitution")`.

4. **Multiple Citations in One Query:**
   - Dual citations across different acts: `"302 PPC and 497 CrPC"` -> extracts both `[("302", "PPC"), ("497", "CrPC")]`.
   - Prefix-style citations: `"Section 302 PPC and Section 497 CrPC"` -> captures both sections and respective acts without cross-contamination.
   - Constitution multi-articles: `"Article 199 and Article 10A"` -> extracts `[("199", "Constitution"), ("10A", "Constitution")]`.
   - Cross-act Article and Section queries: `"Article 199 of Constitution and 302 PPC"` -> extracts both `("199", "Constitution")` and `("302", "PPC")`.
   - Exact channel resolution: `Retriever._exact_channel("302 PPC and 497 CrPC")` maps to both `["ppc-1860-s302", "crpc-1898-s497"]`.
   - Multi-citation search ranking: Both provisions appear in the top 2 retrieved hits and query is verified **NOT** to be falsely refused.

5. **Boundary Conditions:**
   - RRF handling of empty sublists (`rrf([[], []]) == []`) and single-channel results.
   - BM25 Okapi handling of empty token lists and Out-Of-Vocabulary (OOV) tokens without division by zero or errors.

---

## 2. CI Gate Verification (`.github/workflows/ci.yml`)

The retrieval evaluation harness (`python eval.py`) was executed against all 34 adversarial cases (27 gold-labelled cases, 7 adversarial trap cases) under `results/retrieval.json`.

### Evaluation Results Table:
| Configuration | $n$ | Recall@1 | Recall@5 | Recall@10 | Recall@20 | MRR | Refusal Rate | False Refusal | Traps |
|---|---|---|---|---|---|---|---|---|---|
| `sparse_only` | 27 | 0.8519 | 1.0000 | 1.0000 | 1.0000 | 0.9198 | 1.0000 | 0.0370 | 7 |
| `hybrid` | 27 | 0.8519 | 1.0000 | 1.0000 | 1.0000 | 0.9198 | 1.0000 | 0.0370 | 7 |
| **`hybrid+exact`** | **27** | **0.9630** | **1.0000** | **1.0000** | **1.0000** | **0.9815** | **1.0000** | **0.0000** | **7** |

### CI Guardrail Assertions:
```python
r = json.load(open("results/retrieval.json"))["hybrid+exact"]
assert r["recall@1"] >= 0.90     # Result: 0.9630 >= 0.90 -> PASS (+0.0630 margin)
assert r["refusal_rate"] >= 0.85 # Result: 1.0000 >= 0.85 -> PASS (+0.1500 margin)
assert r["false_refusal"] <= 0.15# Result: 0.0000 <= 0.15 -> PASS (-0.1500 headroom)
```

All gates passed. Notice that `false_refusal` dropped to **0.0000** (zero false refusals across all valid adversarial queries), while trap rejection stayed at **100%**.

---

## 3. Resource Utilization & Render Free-Tier Audit

Render's free tier provides **512 MB RAM** and **0.1 CPU**, with automatic spin-down after 15 minutes of inactivity. Under `render.yaml`, `PLB_NO_DENSE=1` disables the local PyTorch sentence-transformer embedding model.

### Resource Utilization Profiling:
- **Cold Boot Import & Initialization Time:**
  - Raw Python import time: **1,583.05 ms** (~1.58 seconds)
  - Full server initialization with route registration: **4,156.19 ms** (~4.1 seconds)
  - Status: Well within Render's 30-second cold-boot timeout window.
- **Memory Footprint:**
  - Tracemalloc Heap Baseline: **22.87 MB**
  - Tracemalloc Heap Peak: **23.07 MB**
  - Initial Process RSS (Resident Set Size): **61.78 MB**
  - Initial Process VMS (Virtual Memory Size): **51.36 MB**
- **Load Test (300 Mixed Endpoints Requests):**
  - Total Duration: **981.64 ms**
  - Average Latency: **3.27 ms/request**
  - Post-Load Process RSS: **66.77 MB**
  - Memory Growth / Leak Delta: **4.99 MB** (stable, flat memory profile)
  - Compliance Target ($< 150$ MB RAM): **PASS** (Utilizes **44.5%** of budget, with **83.23 MB** headroom remaining).

---

## 4. API Verification, Error Handling & Schema Validation

FastAPI server endpoints were enhanced with explicit Pydantic response models (`HealthResponse`, `ChatResponse`, `SourceItem`) and mounted static file serving at `/static`.

### Endpoint Audit:
1. **`GET /api/health`**
   - Verified Response: `{"chunks": 30, "dense": false, "providers": [...], "corpus_verified": false}`
   - Types validated against `HealthResponse` schema.
   - Disallowed methods (e.g. `POST /api/health`) return `405 Method Not Allowed`.

2. **`POST /api/chat`**
   - **Greetings:** Recognized input (`"salam"`, `"hello"`, `"aoa"`) returns 200 with friendly onboarding message, `refused=False`, `sources=[]`, and `provider="assistant"`.
   - **Valid Retrieval:** Returns 200 with answer, `refused=False`, `provider`, and structured `sources` list matching `SourceItem` schema (`citation`, `marginal_note`, `act`, `text`, `score`, `channels`, `verified`, `source_url`).
   - **Deterministic Refusal (Unknown Provision):** Section 999 PPC returns 200 with clear refusal explanation identifying unindexed provision.
   - **Deterministic Refusal (Out of Scope):** Unindexed legal domains (corporate tax, trademark, cyber sovereignty) return 200 with out-of-scope refusal explanation.
   - **Empty / Whitespace Input:** Returns 200 with graceful refusal and `sources=[]`.
   - **Input Validation Errors:** Missing `question` body, invalid types, or malformed JSON payloads return standard `422 Unprocessable Entity` with structured Pydantic error details.
   - Disallowed methods (`GET /api/chat`) return `405 Method Not Allowed`.

3. **Static Endpoints (`GET /`, `GET /research`, `GET /static/...`)**
   - `GET /` serves chat UI (`static/index.html`) with `Content-Type: text/html; charset=utf-8`.
   - `GET /research` serves research dossier (`static/research.html`) with `Content-Type: text/html; charset=utf-8`.
   - `GET /static/banner.png` serves repository asset with `Content-Type: image/png`.
   - Non-existent routes return standard `404 Not Found`.

4. **JSON Schema Generation (`/openapi.json`)**
   - Complete OpenAPI 3.1.0 specifications verified for `ChatRequest`, `ChatResponse`, `HealthResponse`, and `SourceItem`.

---

## 5. Artifacts and Commits Summary
- Modified: `app.py` (Added Pydantic response models `HealthResponse`, `SourceItem`, `ChatResponse`, response model route annotations, static file mounting)
- Modified: `index.py` (Disambiguated Article/Constitution citations, enhanced multi-citation matching, and safe section normalization)
- Modified: `tests/test_core.py` (Expanded from 11 tests to 28 tests for edge queries, whitespace, punctuation, multi-citations, and pure function boundaries)
- Added: `tests/test_api.py` (18 new tests covering API shapes, greetings, refusals, validation errors, static files, and OpenAPI schemas)
- Updated: `results/retrieval.json` (Regenerated benchmark eval metrics: recall@1=0.963, refusal_rate=1.000, false_refusal=0.000)
- Added: `docs/qa_report.md` (Complete audit documentation)
