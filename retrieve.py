"""Stage 4: decide what each question gets to read in C1 / C2 / C2b / C3, and measure retrieval.

This module chooses the EVIDENCE only; how the question is asked (the prompt)
belongs to generate.py. Context files hold no gold labels, so the answer cannot
reach a prompt by accident.

  C1   no documents                                  what the model already knows
  C2   top-k retrieved, in rank order                end-to-end system behaviour
  C2b  ranking with the source removed, first k      key evidence absent
       (= "retrieve k+1, drop the source"; on a retrieval miss it equals C2)
  C3   the source document only (its corpus text)    ceiling: perfect retrieval

Retrieval runs once per question and keeps the top DEPTH documents; every k is
cut from that one list, so the k-sweep cannot differ because search was rerun.

Run:  python retrieve.py --questions 20                 (dev slice, BM25, k in 1 3 5 10)
      python retrieve.py                                (all 500 dev questions)
      python retrieve.py --retriever medcpt             (needs artifacts/medcpt/ from Kaggle)
      python retrieve.py --corpus-limit 2000 ...        (use the corpus slice instead)
"""
import argparse
import json
import time

from corpus import doc_text, load_corpus
from data import ROOT, load_pqal, load_split

DEPTH = 100          # ranking depth saved per question (k <= 10, plus Recall@100 for analysis)
KS = (1, 3, 5, 10)   # k-sweep on dev (CLAUDE.md section 6)
OUT = ROOT / "artifacts" / "retrieval"


def tag(split, retriever=None, n_questions=None, corpus_limit=None):
    """File-name tag; slices are always marked so they can never pass for the full run."""
    parts = [split] + ([retriever] if retriever else [])
    parts += [f"q{n_questions}"] if n_questions else []
    parts += [f"corpus{corpus_limit}"] if corpus_limit else []
    return "_".join(parts)


def rank_questions(retriever, questions, corpus_limit=None):
    """Top-DEPTH [(pmid, score)] per question from the chosen retriever."""
    if retriever == "bm25":
        from index import BM25
        bm25 = BM25(corpus_limit)
        return [bm25.search(q, DEPTH) for q in questions]
    if retriever == "medcpt":
        from medcpt import MedCPT
        return MedCPT(corpus_limit).search_many(questions, DEPTH)
    raise ValueError(f"unknown retriever {retriever!r}")


def contexts_for(condition, source, ranking, k):
    """Document PMIDs given to the model, in order. `ranking` is a list of PMIDs, best first."""
    if condition == "C1":
        return []
    if condition == "C2":
        return ranking[:k]
    if condition == "C2b":
        return [p for p in ranking if p != source][:k]
    if condition == "C3":
        return [source]
    raise ValueError(condition)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:  # binary + "\n": identical bytes on Windows and Linux
        for row in rows:
            f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Retrieve for every question and write C1/C2/C2b/C3 context files.")
    ap.add_argument("--retriever", choices=("bm25", "medcpt"), default="bm25")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--questions", type=int, help="first N questions of the split only (smoke test)")
    ap.add_argument("--corpus-limit", type=int, help="use the corpus slice built by `corpus.py --limit N`")
    args = ap.parse_args()
    if args.split == "test":
        print("NOTE: building TEST contexts. Allowed only for the single final run (CLAUDE.md section 8).")

    records = load_pqal()
    dev, test = load_split(records)
    qids = (dev if args.split == "dev" else test)[: args.questions]
    corpus = load_corpus(args.corpus_limit)

    t0 = time.time()
    rankings = rank_questions(args.retriever, [records[q]["question"] for q in qids], args.corpus_limit)
    print(f"retrieved top-{DEPTH} for {len(qids)} {args.split} questions with {args.retriever} in {time.time() - t0:.0f}s")

    # 1) Rankings, with the rank of the source (None = not in the top DEPTH).
    rows, source_rank = [], {}
    for q, ranked in zip(qids, rankings):
        assert len(ranked) >= max(KS) + 1, f"{q}: only {len(ranked)} documents retrieved"
        pmids = [p for p, _ in ranked]
        source_rank[q] = pmids.index(q) + 1 if q in pmids else None
        rows.append({"qid": q, "source_rank": source_rank[q], "ranking": ranked})
    t = tag(args.split, args.retriever, args.questions, args.corpus_limit)
    write_jsonl(OUT / f"rankings_{t}.jsonl", rows)

    # 2) Context files. C1 and C3 do not depend on the retriever or k, so they are written once.
    def entry(q, condition, pmids, k=None):
        assert all(p in corpus for p in pmids), f"{q}: document missing from corpus"
        return {"qid": q, "condition": condition, "retriever": None if k is None else args.retriever, "k": k,
                "doc_pmids": pmids, "source_in_context": q in pmids, "source_rank": source_rank[q],
                "documents": [corpus[p][1] for p in pmids]}

    base = tag(args.split, None, args.questions, args.corpus_limit)
    for condition in ("C1", "C3"):
        write_jsonl(OUT / f"contexts_{base}_{condition}.jsonl",
                    [entry(q, condition, contexts_for(condition, q, None, None)) for q in qids])
    for k in KS:
        for condition in ("C2", "C2b"):
            ents = [entry(q, condition, contexts_for(condition, q, [p for p, _ in r["ranking"]], k), k)
                    for q, r in zip(qids, rows)]
            write_jsonl(OUT / f"contexts_{t}_k{k}_{condition}.jsonl", ents)

    # 3) Invariants, checked on what was just built.
    for q, r in zip(qids, rows):
        ranking = [p for p, _ in r["ranking"]]
        # C3 is the source exactly as it appears in the corpus, i.e. the stripped PubMedQA context.
        assert corpus[q] == ("L", doc_text(records[q]["contexts"])), f"{q}: C3 text differs from the corpus"
        for k in KS:
            c2, c2b = contexts_for("C2", q, ranking, k), contexts_for("C2b", q, ranking, k)
            assert len(c2) == len(c2b) == k and q not in c2b, f"{q}: bad C2/C2b at k={k}"
            hit = source_rank[q] is not None and source_rank[q] <= k
            assert hit == (q in c2), f"{q}: hit bookkeeping wrong at k={k}"
            if not hit:
                assert c2b == c2, f"{q}: on a miss C2b must equal C2 (k={k})"
            else:
                assert c2b[: source_rank[q] - 1] == c2[: source_rank[q] - 1], f"{q}: C2b reorders documents"

    # 4) Recall@k of the source: how often C2 actually contains the key evidence.
    print(f"\nRecall@k of the source ({args.retriever}, {args.split}, {len(qids)} questions):")
    for k in KS + (DEPTH,):
        hits = sum(1 for q in qids if source_rank[q] is not None and source_rank[q] <= k)
        print(f"  @{k:<3} {hits / len(qids):6.1%}   hits {hits:3d}   misses {len(qids) - hits:3d}")
    print(f"\nwrote {OUT.relative_to(ROOT)}/rankings_{t}.jsonl and contexts_{base}_*.jsonl / contexts_{t}_k*_*.jsonl")
    print("retrieve.py: all checks passed")
