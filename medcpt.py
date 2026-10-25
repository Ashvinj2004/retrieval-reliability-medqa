"""Stage 3, part B: MedCPT dense retrieval (Jin et al., Bioinformatics 2023).

MedCPT has two BERT encoders trained on PubMed search logs: one for queries, one
for articles. Each turns a text into one 768-number vector (the [CLS] state); a
document's score for a question is the dot product of the two vectors.

Choices (from the official model cards and the authors' repository):
  - Articles are encoded as the pair [title, abstract], max 512 tokens. Titles are
    withheld (leakage control 2), so we pass ["", abstract]: the model's own input
    format with an empty title slot. This likely costs MedCPT some accuracy; the
    cost is the same for every document.
  - Queries: max 64 tokens. Similarity: inner product, exact (the authors use
    FAISS IndexFlatIP). Exact search over 273k x 768 is one matrix product, so
    plain numpy gives identical results without the FAISS dependency.

Run:  python medcpt.py --limit 2000       (corpus slice; fine on CPU)
      python medcpt.py                    (full corpus; needs a GPU, e.g. Kaggle)
      python medcpt.py --skip-encode      (reuse saved embeddings, only search)
"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from corpus import load_corpus, stats_path
from data import ROOT, load_pqal, load_split

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
# Pinned model commits: if the authors update the weights, our numbers must not silently move.
REVISION = {ARTICLE_MODEL: "d05a736da4bb84ee4057b7f7999485be6ed85465",
            QUERY_MODEL: "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc"}
ARTICLE_MAX_TOKENS, QUERY_MAX_TOKENS = 512, 64


def emb_dir(limit=None):
    return ROOT / "artifacts" / ("medcpt" if limit is None else f"medcpt_limit{limit}")


def pick_device():
    # FP16 on GPU (T4s have no BF16); FP32 on CPU, where FP16 matmuls are slow or unsupported.
    return ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.float32)


@torch.no_grad()
def encode(model_name, texts, max_tokens, batch_size, pair_with_empty_title):
    """[CLS] embeddings (float32 numpy), in the order of `texts`.

    Batches are formed from texts of similar length so little compute goes on
    padding; the original order is restored before returning.
    """
    device, dtype = pick_device()
    tok = AutoTokenizer.from_pretrained(model_name, revision=REVISION[model_name])
    model = AutoModel.from_pretrained(model_name, revision=REVISION[model_name], torch_dtype=dtype).to(device).eval()
    order = np.argsort([len(t.split()) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        if pair_with_empty_title:
            enc = tok([""] * len(batch), batch, truncation=True, padding=True, max_length=max_tokens, return_tensors="pt")
        else:
            enc = tok(batch, truncation=True, padding=True, max_length=max_tokens, return_tensors="pt")
        cls = model(**enc.to(device)).last_hidden_state[:, 0, :]
        out[idx] = cls.float().cpu().numpy()
        done = start + len(batch)
        if done % (batch_size * 200) < batch_size or done == len(texts):
            rate = done / (time.time() - t0)
            print(f"  {model_name.split('/')[-1]}: {done}/{len(texts)} ({rate:.0f}/s, ~{(len(texts) - done) / rate / 60:.0f} min left)")
    return out


def count_truncated(model_name, texts, max_tokens, pair_with_empty_title):
    """How many texts exceed the encoder's token limit (CLAUDE.md: must be measured and reported)."""
    tok = AutoTokenizer.from_pretrained(model_name, revision=REVISION[model_name])
    n_long = 0
    for start in range(0, len(texts), 4096):
        batch = texts[start:start + 4096]
        enc = tok([""] * len(batch), batch) if pair_with_empty_title else tok(batch)
        n_long += sum(len(ids) > max_tokens for ids in enc["input_ids"])
    return n_long


def build_embeddings(limit=None, batch_size=64):
    corpus = load_corpus(limit)
    pmids = sorted(corpus, key=int)
    texts = [corpus[p][1] for p in pmids]
    n_trunc = count_truncated(ARTICLE_MODEL, texts, ARTICLE_MAX_TOKENS, True)
    print(f"{n_trunc}/{len(texts)} documents ({n_trunc / len(texts):.2%}) exceed {ARTICLE_MAX_TOKENS} tokens and are truncated")
    device, dtype = pick_device()
    t0 = time.time()
    emb = encode(ARTICLE_MODEL, texts, ARTICLE_MAX_TOKENS, batch_size, pair_with_empty_title=True)
    out = emb_dir(limit)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    (out / "pmids.json").write_text(json.dumps(pmids), encoding="utf-8")
    meta = {
        "article_model": ARTICLE_MODEL, "revision": REVISION[ARTICLE_MODEL], "documents": len(pmids), "dim": int(emb.shape[1]),
        "max_tokens": ARTICLE_MAX_TOKENS, "truncated_documents": int(n_trunc),
        "device": device, "dtype": str(dtype).replace("torch.", ""), "seconds": round(time.time() - t0),
        # Provenance: embeddings of an older corpus must refuse to load.
        "corpus_sha256": json.loads(stats_path(limit).read_text(encoding="utf-8"))["sha256"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}: {meta}")


class MedCPT:
    def __init__(self, limit=None):
        d = emb_dir(limit)
        self.emb = np.load(d / "embeddings.npy")
        self.pmids = json.loads((d / "pmids.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        current = json.loads(stats_path(limit).read_text(encoding="utf-8"))["sha256"]
        assert meta["corpus_sha256"] == current, f"{d.name} was built from a different corpus: re-encode it"
        assert self.emb.shape == (len(self.pmids), meta["dim"]), "embeddings and PMID list disagree"

    def search_many(self, questions, k, batch_size=32):
        """Top-k [(pmid, score)] for each question, best first; ties broken by PMID (row order)."""
        q = encode(QUERY_MODEL, questions, QUERY_MAX_TOKENS, batch_size, pair_with_empty_title=False)
        results = []
        for scores in q @ self.emb.T:  # exact inner product with every document
            kth = np.partition(scores, -k)[-k]
            cand = np.flatnonzero(scores >= kth)
            cand = cand[np.lexsort((cand, -scores[cand]))][:k]
            results.append([(self.pmids[i], float(scores[i])) for i in cand])
        return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Encode the corpus with MedCPT and sanity-check retrieval.")
    ap.add_argument("--limit", type=int, help="use the corpus slice built by `corpus.py --limit N`")
    ap.add_argument("--skip-encode", action="store_true", help="reuse saved embeddings")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    if not args.skip_encode:
        build_embeddings(args.limit, args.batch_size)
    dense = MedCPT(args.limit)
    records = load_pqal()
    dev, _ = load_split(records)  # dev only: test is touched once, at the end
    questions = [records[p]["question"] for p in dev]
    n_long_q = count_truncated(QUERY_MODEL, questions, QUERY_MAX_TOKENS, False)
    print(f"{n_long_q}/{len(questions)} dev questions exceed {QUERY_MAX_TOKENS} query tokens")

    found = dense.search_many(questions, 100)
    print(f"\nexample dev question ({dev[0]}): {questions[0]}")
    for rank, (pmid, score) in enumerate(found[0][:5], 1):
        print(f"  {rank}. {pmid:>9}  score={score:7.2f}{'   <- SOURCE' if pmid == dev[0] else ''}")
    ranks = [next((r for r, (p, _) in enumerate(f, 1) if p == pmid), None) for pmid, f in zip(dev, found)]
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / len(ranks) for k in (1, 3, 5, 10, 100)}
    print(f"\ndev Recall@k of the source ({len(dev)} questions): " + "  ".join(f"@{k}={v:.1%}" for k, v in recall.items()))
    print("\nmedcpt.py: all checks passed")
