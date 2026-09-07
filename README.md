# PakLegalBench

Statute retrieval over Pakistani law, plus the evaluation harness that measures
how well it actually retrieves.

The chatbot is the demo. **The measurement is the point.** At least eight
commercial legal AI products serve the Pakistani market and every one advertises
an accuracy figure without publishing evidence. This repo is an attempt at the
first honest number.

## Results

Retrieval over the 34-case adversarial set, seed corpus, BM25 only (no
embeddings installed):

| config | recall@1 | recall@5 | MRR | trap refusal | false refusal |
|---|---|---|---|---|---|
| sparse_only | 0.852 | 0.963 | 0.904 | 1.00 | 0.074 |
| hybrid | 0.852 | 0.963 | 0.904 | 1.00 | 0.074 |
| hybrid+exact | **0.963** | 0.963 | **0.966** | 1.00 | 0.074 |

These are on 30 demo provisions, not the real corpus. They demonstrate the
pipeline works; they are not a finding. Real numbers need `parse.py` and
LEGAL-UQA.

The one thing already worth reporting: the exact-reference channel lifts
recall@1 from 0.852 to 0.963. That is the citation-retrieval failure mode from
the research, reproduced and fixed.

## Live

- **Results and research dossier**: https://hammadshakeelai.github.io/paklegalbench/
- **Demo**: deployed on Render free tier. First request after 15 minutes idle
  takes 30–60 seconds to wake the instance.

The demo runs BM25 + exact reference only. Render's free tier is 512MB RAM and
0.1 CPU, which cannot load an embedding model. Per the table above that config
still reaches recall@1 of 0.963.

## Run it

```bash
pip install -r requirements-dev.txt   # includes torch, for local dense retrieval
cp .env.example .env                  # add one API key
PLB_NO_DENSE=1 uvicorn app:app --reload
```

`requirements.txt` is the deploy set and deliberately excludes torch.
`requirements-dev.txt` adds embeddings, reranking and the eval extras.

Open http://localhost:8000. Without an API key retrieval still works and the
sources panel populates; only generation is disabled.

Drop `PLB_NO_DENSE=1` to enable embeddings (first run downloads ~130MB).

```bash
python eval.py              # adversarial set, no API calls, free
python -m pytest tests/ -q  # 9 tests on the pure functions
python index.py "302 PPC"   # retrieval from the CLI
```

## How retrieval works

Three channels fused with reciprocal rank fusion at k=60:

1. **exact** — regex-matched section and article references
2. **sparse** — BM25, k1=1.5 for long-form legal text
3. **dense** — embeddings, optional

Then an optional cross-encoder rerank.

The exact channel exists because roughly 31% of legal retrieval failures in a
published audit were requests for a specific article reference, where dense
models rank thematically-similar provisions above the one actually named. To an
embedding model `Section 302` and `Section 320` are nearly the same point in
vector space. To a regex they are unrelated. Turn the channel off in the UI to
watch this break.

## Refusal

Two deterministic signals, and there is a story behind them.

The obvious design is a confidence threshold on the fused score. It cannot work.
RRF is rank-based, so the top fused score is always `1/(k+1)` no matter how bad
the match. The first version of this repo shipped that threshold and it silently
never fired. Raw BM25 is no better: `Section 999 PPC` scores *higher* than real
queries because "section", "PPC" and "penalty" all match something.

What actually works:

1. The query names a provision and the exact channel finds no such provision.
   Kills false-premise traps outright.
2. The top hit covers too few of the query's content words.

Both are in `Retriever.refusal_reason()`, and both are measured on every eval
run alongside a **false refusal** rate, because a system that refuses everything
scores 1.00 on trap refusal.

## Corpus

`seed_corpus.json` is **demo data**. The `text` fields are plain-language
descriptions of each provision's effect, not authentic statutory text. Every row
carries `verified: false` and the UI shows a warning banner. Nothing in it may
be cited.

Run `parse.py` to build the real corpus from the published Hugging Face
datasets. That is the step that determines whether any number here means
anything, so read 20 random parsed chunks by hand before trusting it.

## Files

```
index.py             retrieval: BM25, dense, exact, RRF, rerank, refusal
llm.py               Groq -> Gemini -> OpenRouter fallback, grounded prompt
app.py               FastAPI: /api/chat, /api/health
static/index.html    chat UI with per-channel toggles
eval.py              recall@k, MRR, refusal rates. No API calls.
parse.py             real corpus builder (day 1 work)
adversarial.jsonl    34 cases: 27 with gold labels, 7 traps
tests/test_core.py   pure-function tests
```

## Scope and limits

- Statutes only. No case law, no drafting, no case management.
- English only. Urdu retrieval is a later measured question, not a feature.
- The system can tell you what a provision says. It **cannot** tell you whether
  that provision is still good law. Flat retrieval has no amendment history.
- Not legal advice, and not a lawyer. A research retrieval tool with a measured
  error rate.

## Next

1. `parse.py` against the real corpus, then re-run eval
2. Map LEGAL-UQA's 619 pairs onto chunk ids, verify 30 by hand
3. Publish the adversarial set as an open dataset
4. Score end-to-end hallucination using the Stanford definition
