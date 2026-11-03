"""Stage 3: retrieval index over the corpus. Part A: BM25 (CPU).

BM25 is hand-written on a sparse matrix so every line can be defended, and is
verified against the formula computed naively (brute force) on the corpus slice.

  score(q, d) = sum over question terms t of
      IDF(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
  IDF(t)      = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))     (Lucene's always-positive variant)

Every (document, term) weight is precomputed once into a sparse matrix W
(documents x vocabulary), so scoring a question is just summing a few columns.

Run:  python index.py --limit 2000   (index the corpus slice + brute-force check)
      python index.py                (full index + dev Recall@k sanity check)
"""
import argparse
import json
import math
import re
import time
from array import array
from collections import Counter
from functools import lru_cache

import numpy as np
import snowballstemmer
from scipy import sparse

from corpus import load_corpus, stats_path
from data import ROOT, load_pqal, load_split

K1, B = 1.2, 0.75  # textbook values and Lucene's defaults; fixed a priori, not tuned
# Lucene's default English stopword set: frequent words with no topical signal.
STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the their then "
    "there these they this to was will with".split()
)
_WORD = re.compile(r"\w+")  # Unicode-aware: keeps Greek letters (TNF-α and TNF-β are different molecules)
_stemmer = snowballstemmer.stemmer("porter")


@lru_cache(maxsize=None)
def _stem(word):
    # ~1M distinct words but ~65M word occurrences: stem each distinct word only once.
    return _stemmer.stemWord(word)


def tokenize(text):
    """Lowercase, split into words, drop stopwords, Porter-stem. The SAME function for documents and questions."""
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def index_dir(limit=None):
    return ROOT / "artifacts" / ("bm25" if limit is None else f"bm25_limit{limit}")


def build_bm25(limit=None):
    t0 = time.time()
    corpus = load_corpus(limit)
    pmids = sorted(corpus, key=int)  # row i of W is document pmids[i]
    vocab = {}
    # Typed arrays (4 bytes per entry): Python lists of ~35M ints would need ~10x the memory.
    rows, cols, tfs = array("i"), array("i"), array("f")
    assert rows.itemsize == 4, "array('i') must be 32-bit to be read back as int32"
    doc_len = np.zeros(len(pmids))
    for i, pmid in enumerate(pmids):
        counts = Counter(tokenize(corpus[pmid][1]))
        doc_len[i] = sum(counts.values())
        rows.extend([i] * len(counts))
        cols.extend(vocab.setdefault(term, len(vocab)) for term in counts)
        tfs.extend(counts.values())
    print(f"tokenized {len(pmids)} documents ({len(vocab)} distinct terms) in {time.time() - t0:.0f}s")

    rows, cols = np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32)
    tf = np.frombuffer(tfs, dtype=np.float32).astype(np.float64)
    n_docs = len(pmids)
    df = np.bincount(cols, minlength=len(vocab))  # n_t: number of documents containing term t
    idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    avgdl = doc_len.mean()
    length_norm = K1 * (1 - B + B * doc_len / avgdl)  # one value per document
    w = idf[cols] * tf * (K1 + 1) / (tf + length_norm[rows])
    # CSC (column-major): pulling out the handful of columns for a question's terms is fast.
    weights = sparse.csc_matrix((w.astype(np.float32), (rows, cols)), shape=(n_docs, len(vocab)))

    out = index_dir(limit)
    out.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out / "weights.npz", weights, compressed=False)
    (out / "pmids.json").write_text(json.dumps(pmids), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    meta = {
        "k1": K1, "b": B, "documents": n_docs, "vocabulary": len(vocab), "nonzeros": int(weights.nnz),
        "avgdl": round(float(avgdl), 3),
        # Provenance: an index built from an older corpus must refuse to load.
        "corpus_sha256": json.loads(stats_path(limit).read_text(encoding="utf-8"))["sha256"],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"built and saved {out.relative_to(ROOT)} in {time.time() - t0:.0f}s: {meta}")


class BM25:
    def __init__(self, limit=None):
        d = index_dir(limit)
        self.weights = sparse.load_npz(d / "weights.npz").tocsc()
        self.pmids = json.loads((d / "pmids.json").read_text(encoding="utf-8"))
        self.vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        current = json.loads(stats_path(limit).read_text(encoding="utf-8"))["sha256"]
        assert meta["corpus_sha256"] == current, f"{d.name} was built from a different corpus: rebuild it"
        assert (meta["k1"], meta["b"]) == (K1, B), f"{d.name} was built with other BM25 parameters"

    def search(self, query, k):
        """Top-k [(pmid, score)], best first. Ties are broken by PMID so results are fully deterministic."""
        q = Counter(t for t in tokenize(query) if t in self.vocab)
        if not q:
            return []
        # A term repeated in the question counts once per repetition (as in Lucene).
        scores = self.weights[:, list(map(self.vocab.get, q))] @ np.array(list(q.values()), dtype=np.float32)
        kth = np.partition(scores, -k)[-k]  # k-th best score; everything >= it is a candidate (ties included)
        cand = np.flatnonzero((scores >= kth) & (scores > 0))
        cand = cand[np.lexsort((cand, -scores[cand]))][:k]  # score descending, then row (= PMID) ascending
        return [(self.pmids[i], float(scores[i])) for i in cand]


def brute_force_check(bm25, corpus, questions, k=10):
    """Recompute BM25 for EVERY document with plain Python, straight from the formula, and compare."""
    doc_tf = {p: Counter(tokenize(text)) for p, (_, text) in corpus.items()}
    n_docs = len(doc_tf)
    avgdl = sum(sum(c.values()) for c in doc_tf.values()) / n_docs
    for question in questions:
        q = tokenize(question)
        df = {t: sum(1 for c in doc_tf.values() if t in c) for t in set(q)}
        slow = {}
        for pmid, c in doc_tf.items():
            dl = sum(c.values())
            s = sum(math.log(1 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                    * c[t] * (K1 + 1) / (c[t] + K1 * (1 - B + B * dl / avgdl)) for t in q if c[t])
            if s > 0:
                slow[pmid] = s
        slow_top = sorted(slow, key=lambda p: (-slow[p], int(p)))[:k]
        fast = bm25.search(question, k)
        assert [p for p, _ in fast] == slow_top, f"ranking differs for: {question}"
        for p, s in fast:
            assert math.isclose(s, slow[p], rel_tol=1e-4), f"score differs for {p}: {s} vs {slow[p]}"
    print(f"brute-force check: identical top-{k} and scores (rel. tol 1e-4) for {len(questions)} questions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build and sanity-check the BM25 index.")
    ap.add_argument("--limit", type=int, help="index the corpus slice built by `corpus.py --limit N`")
    args = ap.parse_args()

    build_bm25(args.limit)
    bm25 = BM25(args.limit)
    records = load_pqal()
    dev, _ = load_split(records)  # every check here uses dev only: test is touched once, at the end

    if args.limit is not None:  # the naive recomputation is only affordable on the slice
        brute_force_check(bm25, load_corpus(args.limit), [records[p]["question"] for p in dev[:20]])

    first = records[dev[0]]
    print(f"\nexample dev question ({first['pmid']}): {first['question']}")
    for rank, (pmid, score) in enumerate(bm25.search(first["question"], 5), 1):
        print(f"  {rank}. {pmid:>9}  score={score:6.2f}{'   <- SOURCE' if pmid == first['pmid'] else ''}")

    # Sanity check only; the formal Recall@k analysis belongs to stage 4.
    t0 = time.time()
    ranks = []
    for pmid in dev:
        found = [p for p, _ in bm25.search(records[pmid]["question"], 100)]
        ranks.append(found.index(pmid) + 1 if pmid in found else None)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / len(ranks) for k in (1, 3, 5, 10, 100)}
    print(f"\ndev Recall@k of the source ({len(dev)} questions, {time.time() - t0:.0f}s): "
          + "  ".join(f"@{k}={v:.1%}" for k, v in recall.items()))
    print("\nindex.py: all checks passed")
