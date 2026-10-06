"""Stage 1: load PubMedQA PQA-L, freeze the official dev/test split, check the majority baseline.

Run:  python data.py

The first run downloads the official test PMIDs and writes data/split.json
(commit that file). Every later run reads it and re-validates it; it never
re-splits, so the committed file IS the freeze.
"""
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from metrics import LABELS, accuracy, macro_f1

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "data" / "split.json"

# Pinned dataset commit: if the owner edits the dataset, our numbers must not silently move.
# corpus.py loads PQA-U / PQA-A from the same pinned revision.
PUBMEDQA_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"

# The 500 test PMIDs (with gold labels) published by the PubMedQA authors.
OFFICIAL_TEST_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"

# SHA-256 of the sorted test PMIDs in data/split.json. Size and label checks cannot
# catch a dev/test swap of two same-label questions; this can. Changing the split
# therefore also requires editing this line, which makes it a visible, deliberate act.
SPLIT_FINGERPRINT = "d989053b9b6b579d3916378d49a3a0127118ca0cca5952a2209452ab20d2433c"

# Facts from Jin et al. (2019). If the loaded data disagrees, the data is wrong, not the paper.
EXPECTED_COUNTS = {"yes": 552, "no": 338, "maybe": 110}
EXPECTED_SPLIT_COUNTS = {"yes": 276, "no": 169, "maybe": 55}  # each half of the official split


def load_pqal():
    """Return PQA-L as {pmid: record}, validated against the paper's published facts."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train", revision=PUBMEDQA_REVISION)
    records = {}
    for row in ds:
        # PMIDs are strings everywhere: JSON keys are always strings, so a single
        # type avoids silent int-vs-str mismatches when joining with the corpus later.
        pmid = str(row["pubid"])
        assert pmid not in records, f"duplicate PMID {pmid}"
        rec = {
            "pmid": pmid,
            "question": row["question"],                 # the article's title, phrased as a question
            "contexts": row["context"]["contexts"],      # abstract WITHOUT its conclusion = C3 gold context
            "section_labels": row["context"]["labels"],  # e.g. BACKGROUND, METHODS, RESULTS
            "long_answer": row["long_answer"],           # the conclusion: never shown to the model
            "label": row["final_decision"],
        }
        assert rec["label"] in LABELS, f"{pmid}: unexpected label {rec['label']!r}"
        assert rec["question"].strip() and rec["long_answer"].strip(), f"{pmid}: empty question or conclusion"
        assert rec["contexts"] and len(rec["contexts"]) == len(rec["section_labels"]), f"{pmid}: bad contexts"
        # Leakage control 1 (answer leak) for C3: the conclusion must not sit inside the context.
        assert rec["long_answer"].strip() not in " ".join(rec["contexts"]), f"{pmid}: conclusion inside context"
        records[pmid] = rec

    counts = Counter(r["label"] for r in records.values())
    assert counts == EXPECTED_COUNTS, f"label counts {dict(counts)} != {EXPECTED_COUNTS}"
    return records


def freeze_split(records):
    """One-time: fetch the official test PMIDs, cross-check their labels, write data/split.json."""
    with urllib.request.urlopen(OFFICIAL_TEST_URL) as resp:
        official = json.loads(resp.read().decode("utf-8"))
    # Two independent sources (the authors' GitHub vs HuggingFace) must agree on every label.
    for pmid, label in official.items():
        assert pmid in records, f"official test PMID {pmid} missing from PQA-L"
        assert records[pmid]["label"] == label, f"{pmid}: GitHub says {label}, HuggingFace says {records[pmid]['label']}"
    split = {
        "source": OFFICIAL_TEST_URL,
        "dev": sorted((p for p in records if p not in official), key=int),
        "test": sorted(official, key=int),
    }
    SPLIT_PATH.parent.mkdir(exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(f"froze split -> {SPLIT_PATH.relative_to(ROOT)}")


def split_fingerprint(test_ids):
    return hashlib.sha256(",".join(sorted(test_ids, key=int)).encode("utf-8")).hexdigest()


def load_split(records):
    """Read the frozen split and verify it is still a clean, stratified 500/500 partition."""
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    dev, test = split["dev"], split["test"]
    assert len(dev) == len(test) == 500, f"sizes: dev={len(dev)} test={len(test)}"
    assert not set(dev) & set(test), "dev and test overlap"
    assert set(dev) | set(test) == set(records), "split does not cover PQA-L exactly"
    for name, ids in (("dev", dev), ("test", test)):
        counts = Counter(records[p]["label"] for p in ids)
        assert counts == EXPECTED_SPLIT_COUNTS, f"{name} counts {dict(counts)} != {EXPECTED_SPLIT_COUNTS}"
    assert split_fingerprint(test) == SPLIT_FINGERPRINT, f"{SPLIT_PATH.name} test PMIDs differ from the frozen split"
    return dev, test


def majority_baseline(records, dev, test):
    """Always predict the most common dev label: the floor every condition must beat."""
    # Chosen from dev only, so even this trivial model uses no test information.
    majority = Counter(records[p]["label"] for p in dev).most_common(1)[0][0]
    print(f"\nmajority baseline (always '{majority}', chosen on dev):")
    for name, ids in (("dev", dev), ("test", test), ("all", sorted(records, key=int))):
        gold = [records[p]["label"] for p in ids]
        pred = [majority] * len(gold)
        acc, mf1 = accuracy(gold, pred), macro_f1(gold, pred)
        print(f"  {name:<4} n={len(ids):<5} accuracy={acc:.2%}  macro-F1={mf1:.4f}")
        # CLAUDE.md hard rule: if this fails, the data did not load correctly. Stop and fix that first.
        assert abs(acc - 0.5520) < 5e-5 and abs(mf1 - 0.2371) < 5e-5, "majority baseline != 55.20% / 0.2371"


if __name__ == "__main__":
    records = load_pqal()
    print(f"loaded PQA-L: {len(records)} questions, labels {dict(Counter(r['label'] for r in records.values()))}")
    if not SPLIT_PATH.exists():
        freeze_split(records)
    dev, test = load_split(records)
    print(f"split: dev={len(dev)}, test={len(test)}, each 276 yes / 169 no / 55 maybe, disjoint")
    majority_baseline(records, dev, test)
    print("\ndata.py: all checks passed")
