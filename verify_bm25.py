"""Check the hand-written BM25 in index.py against an independent implementation (bm25s, Lucene variant).

Both get the same tokens and the same corpus slice, so any difference comes from
the scoring itself. Expected: identical top-10 sets for every dev question, order
differing only where two documents have exactly the same score, and every score
exactly (k1 + 1) times bm25s's (modern Lucene drops that constant factor, which
cannot change a ranking).

Run (after `python corpus.py --limit 2000` and `python index.py --limit 2000`):
      python verify_bm25.py
"""
import math

import bm25s
import numpy as np

from corpus import load_corpus
from data import load_pqal, load_split
from index import B, BM25, K1, tokenize

LIMIT, K = 2000, 10

if __name__ == "__main__":
    corpus = load_corpus(LIMIT)
    ours = BM25(LIMIT)
    lib = bm25s.BM25(method="lucene", k1=K1, b=B)
    lib.index([tokenize(corpus[p][1]) for p in ours.pmids], show_progress=False)  # same row order as ours

    records = load_pqal()
    dev, _ = load_split(records)
    same_set = same_order = 0
    ratios = []
    for pmid in dev:
        idx, sc = lib.retrieve([tokenize(records[pmid]["question"])], k=K, show_progress=False)
        theirs = {ours.pmids[i]: float(s) for i, s in zip(idx[0], sc[0])}
        mine = ours.search(records[pmid]["question"], K)
        mine_order, theirs_order = [p for p, _ in mine], [ours.pmids[i] for i in idx[0]]
        same_set += set(theirs) == set(mine_order)
        same_order += mine_order == theirs_order
        # The only acceptable reason for a different order: the two swapped documents are exactly tied.
        mine_score = dict(mine)
        for a, b in zip(mine_order, theirs_order):
            if a != b:
                assert math.isclose(mine_score[a], mine_score.get(b, -1.0), rel_tol=1e-6), \
                    f"order differs without a tie: {records[pmid]['question']}"
        ratios += [s / theirs[p] for p, s in mine if p in theirs and theirs[p] > 0]

    ratios = np.array(ratios)
    print(f"identical top-{K} sets:  {same_set}/{len(dev)} dev questions")
    print(f"identical top-{K} order: {same_order}/{len(dev)} (any difference is an exact tie, asserted)")
    print(f"score ratio ours/bm25s: {ratios.min():.6f} .. {ratios.max():.6f}  (k1 + 1 = {K1 + 1})")
    assert same_set == len(dev) and np.allclose(ratios, K1 + 1, rtol=1e-5)
    print("verify_bm25.py: hand-written BM25 matches bm25s")
