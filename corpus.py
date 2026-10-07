"""Stage 2: build the retrieval corpus from PubMedQA PQA-L + PQA-U + PQA-A.

One abstract = one document (no chunking), so "was the source retrieved?" has
exactly one answer. Leakage controls enforced here (CLAUDE.md section 4):

  Answer leak -- the 1000 PQA-L articles (the sources of our questions) enter
    WITHOUT their conclusion, exactly the C3 gold context. Every other article
    keeps its full abstract: other papers' conclusions are legitimate evidence.
    Copies of a source published under another PMID are removed: they would carry
    the conclusion back in, and would survive the source's removal in C2b.
  Title leak -- titles are never read. Each question IS its article's title, so
    indexing titles would reduce retrieval to string matching.

Run:  python corpus.py --limit 2000   (smoke test: all of PQA-L + first 2000 rows of PQA-U and PQA-A)
      python corpus.py                (full corpus, ~273.4k documents, a few minutes)
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter, defaultdict

from datasets import load_dataset

from data import PUBMEDQA_REVISION, ROOT, load_pqal

ARTIFACTS = ROOT / "artifacts"
# Small and committed, so a teammate (or Kaggle) can confirm a rebuild is byte-identical.
STATS_PATH = ROOT / "data" / "corpus_stats.json"

MIN_SENT_WORDS = 8         # shorter sentences are too generic to signal copying
COPY_MIN_FRACTION = 0.5    # a copy shares at least half of the source's long sentences...
COPY_MIN_SENTENCES = 2     # ...and at least two, so one boilerplate line can never qualify

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SENT_END = re.compile(r"(?<=[.!?])\s+")


def doc_text(contexts, conclusion=None):
    """The one definition of document text, shared by the corpus and the C3 context.

    Sections are joined into one plain paragraph: no title, no section headers.
    (A "CONCLUSIONS:" header on every other document would single out the source
    as the one whose conclusion is missing.)
    """
    return " ".join(list(contexts) + ([conclusion] if conclusion else []))


def corpus_path(limit=None):
    # A slice gets its own file so it can never be mistaken for the full corpus.
    return ARTIFACTS / ("corpus.jsonl" if limit is None else f"corpus_limit{limit}.jsonl")


def normalise(text):
    """Lowercase, punctuation and spacing collapsed: formatting differences cannot hide a duplicate."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def long_sentences(text):
    return {n for n in (normalise(s) for s in _SENT_END.split(text)) if len(n.split()) >= MIN_SENT_WORDS}


def find_source_copies(originals, docs):
    """Find documents that copy a source article under another PMID.

    originals: {source pmid: ORIGINAL abstract, conclusion included}; docs: {pmid: (subset, text)}.
    Returns ({copy pmid: source pmid}, largest overlap fraction among pairs NOT judged copies).
    The second value documents the gap between copies and merely related papers.
    """
    index, n_long = defaultdict(set), {}
    for src, text in originals.items():
        sents = long_sentences(text)
        n_long[src] = len(sents)
        for s in sents:
            index[s].add(src)
    copies, max_kept = {}, 0.0
    for pmid, (_, text) in docs.items():
        shared = Counter(src for s in long_sentences(text) for src in index.get(s, ()) if src != pmid)
        for src, k in shared.items():
            frac = k / n_long[src]
            if k >= COPY_MIN_SENTENCES and frac >= COPY_MIN_FRACTION:
                assert pmid not in originals, f"source {pmid} copies source {src}: needs a human decision"
                copies[pmid] = src
            else:
                max_kept = max(max_kept, frac)
    return copies, max_kept


def load_corpus(limit=None):
    """{pmid: (subset, text)}, read from the JSONL written by build()."""
    docs = {}
    with open(corpus_path(limit), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["pmid"]] = (d["subset"], d["text"])
    return docs


def build(limit=None):
    t0 = time.time()
    pqal = load_pqal()  # validated in stage 1
    docs = {pmid: ("L", doc_text(r["contexts"])) for pmid, r in pqal.items()}  # conclusion stripped
    raw_rows = {"L": len(docs)}
    for subset, config in (("U", "pqa_unlabeled"), ("A", "pqa_artificial")):
        split = "train" if limit is None else f"train[:{limit}]"
        ds = load_dataset("qiaojin/PubMedQA", config, split=split, revision=PUBMEDQA_REVISION).flatten()
        raw_rows[subset] = len(ds)
        for pmid, contexts, conclusion in zip(ds["pubid"], ds["context.contexts"], ds["long_answer"]):
            pmid = str(pmid)
            # PMIDs are disjoint across subsets in this release; fail rather than silently pick a winner.
            assert pmid not in docs, f"PMID {pmid} appears in two subsets"
            docs[pmid] = (subset, doc_text(contexts, conclusion))
    print(f"loaded {len(docs)} documents {raw_rows} in {time.time() - t0:.0f}s")

    # 1) Answer leak through copies: the same article under another PMID, conclusion intact.
    originals = {p: doc_text(r["contexts"], r["long_answer"]) for p, r in pqal.items()}
    copies, max_kept = find_source_copies(originals, docs)
    for pmid in copies:
        del docs[pmid]

    # 2) Exact duplicates (same abstract, two PMIDs): keep the lowest PMID; a source is never dropped.
    seen, duplicates = {}, {}
    for pmid in sorted(docs, key=int):
        key = normalise(docs[pmid][1])
        if key in seen:
            assert docs[pmid][0] != "L", f"source {pmid} duplicates {seen[key]}"
            duplicates[pmid] = seen[key]
        else:
            seen[key] = pmid
    for pmid in duplicates:
        del docs[pmid]

    # Binary mode with "\n": identical bytes, hence an identical hash, on Windows and Linux.
    path = corpus_path(limit)
    path.parent.mkdir(exist_ok=True)
    sha = hashlib.sha256()
    with open(path, "wb") as f:
        for pmid in sorted(docs, key=int):
            subset, text = docs[pmid]
            line = (json.dumps({"pmid": pmid, "subset": subset, "text": text}, ensure_ascii=False) + "\n").encode("utf-8")
            sha.update(line)
            f.write(line)
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.0f} MB) in {time.time() - t0:.0f}s")

    # Post-conditions, checked on the file as read back from disk -- the thing later stages use.
    corpus = load_corpus(limit)
    assert corpus == docs, "corpus file does not round-trip"
    for pmid, r in pqal.items():
        assert corpus[pmid] == ("L", doc_text(r["contexts"])), f"source {pmid} missing or not conclusion-stripped"
    assert not find_source_copies(originals, corpus)[0], "copies of source articles remain"

    words = sorted(len(text.split()) for _, text in corpus.values())
    pct = lambda q: words[min(len(words) - 1, int(q * len(words)))]
    return {
        "pubmedqa_revision": PUBMEDQA_REVISION,
        "limit": limit,
        "raw_rows": raw_rows,
        "removed_source_copies": {p: copies[p] for p in sorted(copies, key=int)},  # copy -> source
        "max_overlap_kept": round(max_kept, 3),
        "removed_exact_duplicates": {p: duplicates[p] for p in sorted(duplicates, key=int)},  # dropped -> kept
        "documents": len(corpus),
        "documents_by_subset": dict(Counter(subset for subset, _ in corpus.values())),
        "words_per_document": {"min": words[0], "median": pct(0.5), "p95": pct(0.95), "p99": pct(0.99), "max": words[-1]},
        "sha256": sha.hexdigest(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the PubMedQA retrieval corpus.")
    ap.add_argument("--limit", type=int, help="first N rows of PQA-U and PQA-A only (PQA-L is always complete)")
    args = ap.parse_args()
    stats = build(args.limit)
    stats_path = STATS_PATH if args.limit is None else corpus_path(args.limit).with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    summary = dict(stats, removed_exact_duplicates=f"{len(stats['removed_exact_duplicates'])} (listed in stats file)")
    print(json.dumps(summary, indent=1))
    print(f"stats -> {stats_path.relative_to(ROOT)}\ncorpus.py: all checks passed")
