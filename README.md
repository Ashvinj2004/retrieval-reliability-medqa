# Does Retrieval Make Medical QA More Reliable?

A controlled evaluation of an open 8B language model on PubMedQA, with and
without retrieval. The same model answers the same 1,000 yes/no/maybe questions
under four conditions, and we measure accuracy, calibration, confident errors
and whether retrieval found the source article.

| Condition | The model reads |
|---|---|
| C1 | the question only (closed-book) |
| C2 | the question + top-k abstracts retrieved from a 273,441-abstract corpus |
| C2b | the same retrieval with the question's own source article removed |
| C3 | the question + its source abstract, conclusion removed (ceiling) |

The full design, leakage controls and pre-registered decision rules are in
`RESEARCH_PLAN.md`.

## Files, in pipeline order

| File | Purpose |
|---|---|
| `data.py` | load PubMedQA PQA-L; freeze the official 500 dev / 500 test split; majority baseline check (55.20% / 0.2371) |
| `corpus.py` | build the retrieval corpus; remove answer leaks (source conclusions, copies, duplicates) and titles |
| `index.py` | BM25 retrieval, implemented on a sparse matrix |
| `verify_bm25.py` | checks our BM25 against the `bm25s` library |
| `medcpt.py` | MedCPT dense retrieval with exact inner-product search |
| `retrieve.py` | builds the C1 / C2 / C2b / C3 inputs; Recall@k of the source |
| `generate.py` | asks the model; reads its probability for yes / no / maybe |
| `probe.py` | memorisation probe (verbatim recall of conclusions) |
| `metrics.py` | all metrics and statistics, hand-written, with self-tests |
| `evaluate.py` | RQ1-RQ3 analysis: McNemar tests, bootstrap CIs, calibration, risk-coverage |
| `report.py` | the paper's tables and figures |
| `run_all.py` | runs the whole study in one command |
| `data/split.json` | the frozen dev/test split |
| `data/corpus_stats.json` | corpus statistics and SHA-256, to verify a rebuild |

## How to run

```
python -m venv .venv
.venv\Scripts\activate          (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python metrics.py               self-test of every metric
python run_all.py --smoke       whole pipeline on CPU: tiny model, corpus slice, 20 questions
```

The full study needs a GPU (we use Kaggle's 2x T4): see `RUN_ME.md`, or
`KAGGLE.md` for running each step by hand. Every expensive step has a `--limit`
or `--questions` flag for small sanity checks.
