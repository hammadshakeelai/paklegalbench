# CLAUDE.md

Context for Claude Code working in this repo. Read this before changing anything.

## What this project is

A statute retrieval system over Pakistani law, plus the evaluation harness that
measures it. Eight commercial products already serve this market and all
advertise accuracy figures without evidence. This repo exists to produce the
first honest number.

**The measurement is the product. The chatbot is the demo.**

If a change would improve the demo but not the measurement, it is low priority.
If a change would make the numbers look better without making retrieval better,
it is not acceptable.

## Decisions already made — do not relitigate

These were settled during research. Reopening them wastes tokens.

- **Statutes only.** Constitution, PPC, CrPC. No case law, no drafting, no case
  management, no lawyer marketplace. Scope creep kills the one defensible claim.
- **English only** for now. Urdu is a later measured question, not a feature.
- **No vector database.** At 20k–50k chunks, NumPy brute force is exact and
  takes milliseconds. Qdrant/Chroma would add ops complexity and nothing else.
- **Embeddings run locally**, never through an API. Re-embedding after a
  chunking change must be free and unlimited, because iterating on chunking is
  what improves retrieval.
- **Three channels fused with RRF at k=60**: exact reference, BM25, dense.
  Channel weights are `exact 1.4 / sparse 1.0 / dense 0.8`, in
  `RetrievalConfig.weights`. The rationale for those three values is not
  recorded anywhere in the repo; if you change them, record why.
- **BM25 k1=1.5**, higher than the 1.2 default, because term repetition carries
  real signal in long-form legal text.

## The two findings this repo is built around

**1. Dense retrieval fails on citations.** In a published audit of a legal RAG
system, 31% of failed retrievals were requests for a specific article reference,
because the embedding model ranked thematically-similar provisions above the one
named. To an embedding model `Section 302` and `Section 320` are nearly the same
vector. The `exact` channel in `index.py` is the fix, and it lifts recall@1 from
0.852 to 0.963 on the current set.

**2. You cannot threshold on the fused score.** This bug already shipped once.
RRF is rank-based, so the top fused score is always `1/(k+1)` no matter how bad
the match. A confidence threshold on it silently never fires. Raw BM25 is worse:
`Section 999 PPC` outscores real queries because "section", "PPC" and "penalty"
all match something.

Refusal uses two deterministic signals instead, in `Retriever.refusal_reason()`:
- the query names a provision the exact channel cannot find
- the top hit covers too few of the query's content words
  (`min_term_coverage`, currently 0.34)

**If you touch refusal logic, run `python eval.py` and check BOTH
`refusal_rate` and `false_refusal`.** A system that refuses everything scores a
perfect 1.00 on the first one.

These are not soft targets. `.github/workflows/ci.yml` fails the build on the
`hybrid+exact` row of `results/retrieval.json`:

| gate | threshold | seed-corpus snapshot |
|---|---|---|
| `refusal_rate` | `>= 0.85` | 1.00 |
| `false_refusal` | `<= 0.15` | 0.074 |
| `recall@1` | `>= 0.90` | 0.963 |

The snapshot column drifts; `results/retrieval.json` is authoritative. On those
values `false_refusal` has the least headroom. Note what the gate does *not*
cover: `mrr`, `recall@5..20`, and every config other than `hybrid+exact` can
regress and still go green.

## Working rules

- **Write tests for pure functions before wiring anything up.** `rrf`,
  `tokenize`, `extract_references` and the metric functions are deterministic.
  Testing them is cheap and it is what keeps us out of screenshot-debug loops.
- **Run `python eval.py` after any retrieval change.** It makes no API calls, so
  it is free and unlimited. Numbers before features, always.
- **`Chunk.indexed_text()` is the index.** Both BM25 and the embedding model see
  `act_short + section_label + marginal_note + chapter + text`, not the raw
  provision. The prepended header exists because a bare paragraph of statute
  never says which act it belongs to. Changing that method changes every number
  in `results/` and in the published dossier — re-run eval and say so.
- **`STOPWORDS` in `index.py` is load-bearing for refusal, not just for BM25.**
  Term coverage is computed over non-stopword query terms, so adding words
  shrinks the denominator, raises coverage, and fires fewer refusals — moving
  `false_refusal` and trap `refusal_rate` together, one for the better and one
  for the worse. The list is deliberately short for exactly this reason.
- **Never invent statutory text.** `seed_corpus.json` contains 30 paraphrased
  demo provisions (13 Constitution, 9 PPC, 8 CrPC) with `verified: false` on
  every row and a warning banner in the UI. Writing plausible-looking fake
  statute into a tool built to measure hallucination would be self-defeating. If
  real text is needed, it comes from `parse.py`.
- **Never claim a provision is current, unamended, or good law.** Flat retrieval
  has no amendment history. This limitation is disclosed, not hidden. It is also
  rule 4 of `SYSTEM_PROMPT` in `llm.py`; keep the two in sync.
- Keep `index.py` free of framework imports. It must stay runnable as
  `python index.py "302 PPC"`.

## Commands

```bash
pip install -r requirements-dev.txt   # local, includes torch
PLB_NO_DENSE=1 uvicorn app:app --reload
python eval.py                        # free, no API calls
python -m pytest tests/ -q
python index.py "302 PPC"             # retrieval from CLI
```

```bash
python -m pytest tests/test_core.py::test_rrf_weights_applied -q   # single test
python eval.py --adversarial                # explicit; also the no-flag default
python eval.py --legal-uqa                  # needs `datasets` + network
python eval.py --out results/experiment.json
python parse.py --out chunks.json           # real corpus, not yet wired in
```

`PLB_NO_DENSE=1` skips loading the embedding model. **It is read only by
`app.py`.** `eval.py` and `python index.py` build the retriever directly and
always attempt the dense channel, so `PLB_NO_DENSE=1 python eval.py` does
nothing — with `sentence-transformers` installed, eval pays the model load
anyway. To run those without dense, uninstall it or pass `load_dense=False`.

## How the pieces fit

`index.py` is the whole retrieval system and depends on nothing else in the
repo. `app.py` and `eval.py` are both thin consumers of it, which is what makes
the eval harness trustworthy: it exercises the same `Retriever.search()` and
`Retriever.should_refuse()` that the server calls.

**Graceful degradation is a contract, not a convenience.** `DenseIndex` and
`Reranker` catch `ImportError` in `__init__` and set `available = False`; every
caller checks. A missing `sentence-transformers` must always mean "sparse +
exact only", never a crash — that is the exact configuration Render runs and the
one the published headline number describes.

**What `eval.py` actually measures.** `score()` overrides `top_k` and
`candidates` to 20 regardless of the config passed in, so `recall@20` is the
ceiling of the candidate pool, not of the app's default `top_k=5`. Trap cases
(`expect_refusal: true`) `continue` before any recall accumulation, so `n` is
27, not 34, and traps are scored only through `refusal_rate`. The `dense_only`
row is skipped entirely when dense is unavailable, which is why the published
table has three rows.

**Both committed JSONs are build outputs.** `results/retrieval.json` is written
by `eval.py`; `pages.yml` re-runs eval on every push to main and copies the
result to `docs/retrieval.json`. The published page therefore always matches the
code, even when the checked-in copies have drifted locally.

## Gotchas

- `should_refuse(hits, config, query="")` takes `query` as the **third**
  positional parameter with an empty default. Call it without a query and the
  missing-reference branch silently never fires — the same class of bug as the
  fused-score threshold. Prefer `refusal_reason()`, which returns the reason
  string and takes the query first.
- `load_chunks()` is called with no argument in `app.py`, `eval.py` and
  `build_default()`, so the corpus path is hardcoded to `seed_corpus.json`.
  `parse.py` writes `chunks.json` (gitignored) and nothing reads it. Threading
  that path through is part of next-work item 1, not a separate afterthought.
- `static/research.html` and `docs/research.html` are byte-identical copies.
  Edit one and the other drifts. (`static/index.html` is the chat UI and
  `docs/index.html` is the Pages landing page — those two are genuinely
  different files, not copies.)
- LLM model IDs on free tiers move. Override with `GROQ_MODEL`, `GEMINI_MODEL`,
  `OPENROUTER_MODEL` rather than editing `llm.py`.

## Layout

```
index.py                     retrieval: BM25, dense, exact, RRF, rerank, refusal
llm.py                       Groq -> Gemini -> OpenRouter fallback, grounded prompt
app.py                       FastAPI: /api/chat, /api/health, /, /research
eval.py                      recall@k, MRR, refusal rates
parse.py                     real corpus builder
seed_corpus.json             30 demo provisions, all verified: false
adversarial.jsonl            34 cases: 27 gold-labelled, 7 traps
tests/test_core.py           9 tests on tokenize / extract_references / rrf
results/retrieval.json       eval output, regenerated by CI
static/                      chat UI with per-channel toggles + research page
docs/                        GitHub Pages: research dossier + results
.github/workflows/ci.yml     tests, eval, refusal regression gate
.github/workflows/pages.yml  re-runs eval, publishes docs/
requirements.txt             deploy set, no torch
requirements-dev.txt         adds sentence-transformers, datasets, pytest
```

## Deployment

- **GitHub Pages** serves `docs/` — the research dossier and results. Static,
  always on. `pages.yml` regenerates `docs/retrieval.json` from a fresh eval run
  on every push to main, so the published numbers cannot go stale.
- **Render free tier** serves the app, via `render.yaml`. 512MB RAM and 0.1 CPU,
  so `PLB_NO_DENSE=1` is mandatory there; torch will OOM the instance. Free
  services also spin down after 15 minutes with a 30–60s cold start.
- `requirements.txt` is deploy-only and deliberately excludes torch.
  `requirements-dev.txt` has the ML extras. Do not merge them.
- API keys are set in the Render dashboard (`sync: false`), never in
  `render.yaml`. Without a key retrieval still works and the sources panel
  populates; only generation is disabled.

## Next work, in order

1. `parse.py` against the real Hugging Face datasets
   (`AyeshaJadoon/Pakistan_Laws_Dataset`). Read 20 random parsed chunks by hand
   before trusting anything downstream. Pakistani acts are inconsistently
   formatted and amendment footnotes are where parsers break. Note that nothing
   currently loads `chunks.json` — see Gotchas.
2. Map LEGAL-UQA's 619 pairs onto chunk ids. `load_legal_uqa()` in `eval.py`
   currently regexes an article number out of the context field and drops
   whatever it cannot match. Verify 30 mappings by hand. A sloppy mapping
   silently invalidates every published number.
3. Publish `adversarial.jsonl` as an open dataset.
4. Score end-to-end hallucination using the Stanford definition, so the figure
   is comparable to the published 17% and 33%.

## Tone in the README and any write-up

State the numbers plainly, including unflattering ones. The entire value of this
project rests on being the one source in this space that publishes its method.
Do not add marketing language, and do not round anything up.
