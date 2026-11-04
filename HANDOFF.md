# Handoff: running the Llama experiments

This file takes over from the person who built the pipeline. Everything up to
the model runs is finished and tested. What is left needs a GPU and access to
Llama, which is what this handoff covers. Read it once fully before starting.

---

## 1. Where the project stands

**The study (one paragraph).** We test whether letting an AI model look things
up in medical papers (retrieval, "RAG") makes it more *reliable*, not just more
accurate. One model (Llama-3.1-8B-Instruct, never retrained) answers 1,000
PubMedQA yes/no/maybe questions under four conditions:

| Condition | The model reads | Shows |
|---|---|---|
| C1 | the question only | what it already knows |
| C2 | question + top-k abstracts found by search | what a real system does |
| C2b | same search, the question's own paper removed | whether confidence drops when evidence is missing |
| C3 | question + its own paper, conclusion removed | the best case |

The full plan, leakage controls and rules are in `RESEARCH_PLAN.md`. Read
sections 2, 4, 6 and 8 at least.

**Done and tested (all on `main`):**

| File | What it does | Key check |
|---|---|---|
| `data.py` | loads the 1,000 questions, official 500 dev / 500 test split | always-"yes" = 55.20% / macro-F1 0.2371 |
| `corpus.py` | search library of 273,441 abstracts, answer leaks removed | SHA-256 in `data/corpus_stats.json` |
| `index.py` | keyword search (BM25), hand-written | dev Recall@5 = 84.6% |
| `verify_bm25.py` | checks our BM25 against the `bm25s` library | same top-10 for all 500 dev questions |
| `medcpt.py` | medical neural search (MedCPT) | tested on a 5,000-document slice only |
| `retrieve.py` | builds the reading material for C1 / C2 / C2b / C3 | invariants asserted for every question |
| `generate.py` | asks Llama, reads its probability for yes / no / maybe | tested with a tiny model only |
| `evaluate.py` | all metrics and statistics for RQ1-RQ3 | tested with a tiny model only |
| `probe.py` | checks whether Llama memorised the papers' conclusions | tested with a tiny model only |
| `metrics.py` | every metric, written by hand | `python metrics.py` runs a self-test |

**Not done yet:** every Llama run. There are no real results yet.

**Decided in advance (do not change after seeing results; `RESEARCH_PLAN.md` section 6):**
- Both search methods (BM25 and MedCPT) are run and reported.
- k = the value in {1, 3, 5, 10} with the best dev C2 macro-F1, per search method; ties go to the smaller k. `evaluate.py --sweep` applies this automatically.
- Confident-error threshold tau = 0.9, with 0.8 and 0.95 reported as a check.
- Prompt version v1 stays unless the 20-question check (step D) fails.
- **The 500 test questions are run once, at the very end.** Never run anything on test before the dev analysis is finished and agreed by the team.

---

## 2. Step by step

### A. Get access to Llama (do this first; approval can take hours or days)

1. Create or sign in to an account at **huggingface.co**.
2. Open **huggingface.co/meta-llama/Llama-3.1-8B-Instruct**, fill in the access
   form (use your university as affiliation) and accept the license.
3. Wait for the approval email (status is also shown on that page).
4. When approved: **Settings → Access Tokens → Create new token**, type **Read**,
   name it `kaggle`. Copy it. Never paste it into a notebook cell, chat, file
   or GitHub; it goes only into a Kaggle secret (step B4).

### B. Set up Kaggle (free GPUs: 30 hours per week)

1. Sign in at **kaggle.com** → **Settings → Phone verification**. Without it,
   GPUs and internet stay off.
2. **Create → New Notebook.**
3. Right-hand panel → **Session options**: **Accelerator: GPU T4 x2**, **Internet: On**.
4. **Add-ons → Secrets → Add secret**: label `HF_TOKEN`, value = the token from A4.
   Make sure its switch is **on** for this notebook.

### C. Paste these cells (delete the sample cell first; one block = one cell)

**Cell 1: our code and the data**
```
!git clone https://github.com/Ashvinj2004/retrieval-reliability-medqa.git
%cd retrieval-reliability-medqa
!pip install -q snowballstemmer accelerate
!python data.py
!python corpus.py
!git diff --stat data/corpus_stats.json
```

**Cell 2: both search methods and the reading material**
```
!python index.py
!python retrieve.py
!python medcpt.py --batch-size 128
!python retrieve.py --retriever medcpt
```

**Cell 3: the token**
```
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

**Cell 4: 20-question format check**
```
!python generate.py --contexts artifacts/retrieval/contexts_dev_C3.jsonl --limit 20
!python -c "import json,glob; r=[json.loads(l) for l in open(glob.glob('artifacts/generations/*contexts_dev_C3.jsonl')[0])]; print('mean label_mass', sum(x['label_mass'] for x in r)/len(r))"
```

**Cell 5: the long runs**
```
!for f in artifacts/retrieval/contexts_dev_C1.jsonl artifacts/retrieval/contexts_dev_C3.jsonl artifacts/retrieval/contexts_dev_*_k*_C2*.jsonl; do python generate.py --contexts $f; done
!python probe.py
!zip -r -q results.zip artifacts/generations artifacts/probe artifacts/medcpt/meta.json
!ls -lh results.zip
```

### D. Run cells 1-4 interactively and check every line below

Run each cell with **Shift+Enter**, one at a time.

| After | Must see | If not |
|---|---|---|
| Cell 1 | `data.py: all checks passed` and `55.20%` / `0.2371` | stop: data did not load correctly |
| Cell 1 | `corpus.py: all checks passed`, `"documents": 273441` | stop |
| Cell 1 | **`git diff` prints nothing** | **stop**: Kaggle built a different corpus; results would not match ours |
| Cell 2 | `index.py` shows `@5=84.6%` and `all checks passed` | stop |
| Cell 2 | `medcpt.py` prints a truncation line, a Recall line and `all checks passed` (15-30 min) | stop |
| Cell 2 | both `retrieve.py` runs end with `all checks passed` | stop |
| Cell 3 | no output, no error | check the secret name is exactly `HF_TOKEN` and switched on |
| Cell 4 | `mean label_mass` **above 0.95** | **stop**: the prompt needs fixing before any long run (team decision, dev only) |

Write down the numbers from the MedCPT lines (truncation share and dev Recall@k):
the paper needs them.

### E. Run everything in the background (several hours)

Cell 5 makes about 9,000 Llama calls plus the probe. An interactive session can
time out if the browser closes, so:

1. **Save Version** (top right) → **Save & Run All (Commit)** → **Save**.
2. Kaggle reruns cells 1-5 in the background (limit 12 hours). The browser can be closed.
3. Follow progress in the notebook's **Versions** list. Expect roughly 4-6 GPU hours.

### F. Bring the results back

1. When the version is **Complete**: open it → **Output** tab → download `results.zip`.
2. Share it with the team (it is a few MB; do NOT commit it to GitHub; the
   `artifacts/` folder is git-ignored on purpose).

---

## 3. After the results: the dev analysis (on any laptop, no GPU)

**One-time local setup**
```
git clone https://github.com/Ashvinj2004/retrieval-reliability-medqa.git
cd retrieval-reliability-medqa
python -m venv .venv
```
Activate it (Windows: `.venv\Scripts\activate`, Mac/Linux: `source .venv/bin/activate`), then:
```
pip install -r requirements.txt
python metrics.py
python data.py
```
`requirements.txt` was pinned on Python 3.14. If a pin will not install on your
Python, install that package without the version and note the version you got.

Unzip `results.zip` into the project folder so that `artifacts/generations/`
and `artifacts/probe/` exist.

**Step 1: choose k by the pre-registered rule (once per search method)**
```
python evaluate.py --model Llama-3.1-8B-Instruct --tag dev --retriever bm25 --sweep
python evaluate.py --model Llama-3.1-8B-Instruct --tag dev --retriever medcpt --sweep
```
Each prints `chosen k = ...`. Use exactly that; do not pick by eye.

**Step 2: the full dev analysis at the chosen k**
```
python evaluate.py --model Llama-3.1-8B-Instruct --tag dev --retriever bm25 --k <chosen>
python evaluate.py --model Llama-3.1-8B-Instruct --tag dev --retriever medcpt --k <chosen>
```
This prints accuracy, macro-F1, confident errors, calibration (ECE), AURC, the
C2 vs C1 and C3 vs C2 comparisons with confidence intervals, the hit/miss split
(RQ2) and the C2b confidence test (RQ3). A copy is saved in `artifacts/results/`.

**How to read it honestly**
- A difference whose 95% interval includes 0 is **not** a finding. Say "no
  detectable difference", never "slightly better".
- If the script prints "too few misses", the RQ2 comparison is descriptive only.
- If retrieval does not help, that is a valid result; do not change settings to
  make it look better (`RESEARCH_PLAN.md` section 1).
- "maybe" is a real answer, not "I don't know".

**Still to be built after the dev analysis** (do not improvise these):
1. The memorisation analysis: C1 accuracy on questions `probe.py` flags as
   memorised vs the rest.
2. The single final TEST run (`retrieve.py --split test`, then the four
   conditions at the chosen k for both search methods). `evaluate.py`
   deliberately refuses test files until this step is added.

---

## 4. Rules that must not be broken

- The dev/test split in `data/split.json` never changes.
- Test questions are run **once**, at the end, with settings fixed on dev.
- No fine-tuning; the model weights stay fixed.
- Every number in the paper must come from a script in this repo, so it can be re-run.
- Never commit tokens, `results.zip` or anything in `artifacts/`.

## 5. If something goes wrong

| Problem | Fix |
|---|---|
| `401` / `gated repo` error in Cell 4 | Llama access not approved yet, or the `HF_TOKEN` secret is missing or switched off |
| No GPU option, or `git clone` fails | phone verification not done, or Internet is off in the right panel |
| `CUDA out of memory` | check Accelerator is **GPU T4 x2** (not P100), then restart the session |
| `git diff` in Cell 1 prints something | stop and tell the team; do not continue |
| The commit run stops after 12 hours | rerun; files already written are recreated identically |
| Anything else ends without `all checks passed` | stop; the scripts fail on purpose when data looks wrong |
