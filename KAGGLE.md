# Running the GPU steps on Kaggle

Kaggle notebooks are used only as launchers: every cell runs one of our `.py`
files, so no project logic lives in a notebook (CLAUDE.md section 9).

## One-time setup

1. Phone-verify the Kaggle account (Settings). Without it, GPUs and internet stay off.
2. New notebook. In the right panel: **Accelerator: GPU T4 x2**, **Internet: on**.

## Stage 3b: encode the corpus with MedCPT (~15-30 min on a T4)

Run these cells in order:

```
!git clone -b stage3-index https://github.com/Ashvinj2004/retrieval-reliability-medqa.git
%cd retrieval-reliability-medqa
!python data.py
!python corpus.py
!git diff --stat data/corpus_stats.json
!python medcpt.py --batch-size 128
```

What to check in the output:

- `data.py` must end with `all checks passed` and the 55.20% / 0.2371 baseline.
- **`git diff --stat` must print nothing.** That means the corpus Kaggle built is
  byte-identical to ours (same SHA-256). If it prints a change, stop: the
  embeddings would describe a different corpus.
- `medcpt.py` prints the truncation count for the full corpus, a dev Recall@k
  line, and `all checks passed`.

Then download the folder `artifacts/medcpt/` (`embeddings.npy` ~0.84 GB,
`pmids.json`, `meta.json`) from the notebook's Output panel and put it in
`artifacts/medcpt/` on the laptop. Check locally with:

```
python medcpt.py --skip-encode
```

It refuses to load if the embeddings came from a different corpus.

## Stage 5: Llama answers on DEV (prompt and k are chosen here; test is NOT run yet)

Needs approved access to `meta-llama/Llama-3.1-8B-Instruct` on HuggingFace and a
token stored as a Kaggle secret named `HF_TOKEN` (Add-ons > Secrets). Never put
the token in a cell or in the repo.

```
!git clone -b stage5-generate https://github.com/Ashvinj2004/retrieval-reliability-medqa.git
%cd retrieval-reliability-medqa
!pip install -q snowballstemmer
!python corpus.py
!git diff --stat data/corpus_stats.json
!python index.py
!python retrieve.py
```

```
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

First a 20-question check, then the full dev runs:

```
!python generate.py --contexts artifacts/retrieval/contexts_dev_C3.jsonl --limit 20
!for f in artifacts/retrieval/contexts_dev_C1.jsonl artifacts/retrieval/contexts_dev_C3.jsonl artifacts/retrieval/contexts_dev_bm25_k*_C2*.jsonl; do python generate.py --contexts $f; done
```

Check after the 20-question run: `label_mass` should be close to 1 (the model
answers with one of the three words). Download `artifacts/generations/` when done.
