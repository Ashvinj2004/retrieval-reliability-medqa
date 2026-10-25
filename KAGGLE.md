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
