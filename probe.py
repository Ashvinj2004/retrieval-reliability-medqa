"""Memorisation probe: has the model seen these papers' conclusions during training?

PubMedQA abstracts (pre-2019) are very likely in Llama 3's training data. If the
model memorised a paper's conclusion, its closed-book (C1) answer is recall of
the test item, not medical knowledge, which would inflate C1.

Test: give the title and the abstract body (exactly the C3 text), then the cue
"Conclusion:", generate greedily, and compare with the REAL conclusion.
Paraphrase is expected (the body contains the results), so only VERBATIM
reproduction counts:
  leading_match  words matching in a row from the start
  overlap_8gram  share of the real conclusion's 8-word sequences reproduced exactly
Words are lowercased with punctuation removed, so the tokenizer cannot affect the score.

Run:  python probe.py --limit 20 --model HuggingFaceTB/SmolLM2-135M-Instruct   (CPU pipeline test)
      python probe.py                                                          (Llama, on Kaggle; dev)
"""
import argparse
import json
import re
import time

from corpus import doc_text
from data import ROOT, load_pqal, load_split
from generate import DEFAULT_MODEL, load_model

MAX_NEW_TOKENS = 40   # ~30 words: enough to judge an 8-word verbatim run, and keeps the GPU run short
N = 8                 # n-gram length for verbatim overlap; 8 words rarely repeat by chance
MEMORISED_WORDS = 8   # a leading run of >= 8 identical words is counted as memorised

_WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return _WORD.findall(text.lower())


def leading_match(generated, truth):
    n = 0
    for a, b in zip(generated, truth):
        if a != b:
            break
        n += 1
    return n


def overlap_ngrams(generated, truth, n=N):
    """Share of the truth's n-grams that appear verbatim in the generated text (0 if truth is too short)."""
    truth_ngrams = {tuple(truth[i:i + n]) for i in range(len(truth) - n + 1)}
    if not truth_ngrams:
        return 0.0
    gen_ngrams = {tuple(generated[i:i + n]) for i in range(len(generated) - n + 1)}
    return len(truth_ngrams & gen_ngrams) / len(truth_ngrams)


def probe_path(model_name, split, limit=None):
    return ROOT / "artifacts" / "probe" / f"{model_name.split('/')[-1]}_{split}{'_q' + str(limit) if limit else ''}.jsonl"


def run_probe(tok, model, model_name, records, qids, split, limit=None):
    """Probe every question in `qids`; the model is passed in so a caller loads it once. Resumable."""
    import torch

    out = probe_path(model_name, split, limit)
    if out.exists() and sum(1 for _ in open(out, encoding="utf-8")) == len(qids):
        print(f"  already done: {out.name}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    t0, n_mem = time.time(), 0
    with open(tmp, "wb") as f:
        for i, q in enumerate(qids, 1):
            r = records[q]
            prompt = f"Title: {r['question']}\nAbstract: {doc_text(r['contexts'])}\nConclusion:"
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
            generated = tok.decode(gen_ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            gw, tw = words(generated), words(r["long_answer"])
            row = {"qid": q, "leading_match": leading_match(gw, tw), "overlap_8gram": overlap_ngrams(gw, tw),
                   "memorised": leading_match(gw, tw) >= MEMORISED_WORDS, "generated": generated, "model": model_name}
            n_mem += row["memorised"]
            f.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            if i % 100 == 0 or i == len(qids):
                print(f"  probe {split}: {i}/{len(qids)} ({(time.time() - t0) / i:.1f}s per question)")
    tmp.replace(out)
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    mean_overlap = sum(r["overlap_8gram"] for r in rows) / len(rows)
    print(f"  {len(rows)} {split} questions | leading match >= {MEMORISED_WORDS} words: {n_mem} ({n_mem / len(rows):.1%})"
          f" | mean 8-gram overlap {mean_overlap:.1%}")
    return out


if __name__ == "__main__":
    # Self-check of the two scores on a hand-made example, before any model is loaded.
    t = words("Breaks in the cold chain were frequent and compromised vaccine potency in two districts.")
    g = words("Breaks in the cold chain were frequent and compromised the potency of vaccines.")
    assert leading_match(g, t) == 9                      # 'breaks ... compromised' then 'the' != 'vaccine'
    assert abs(overlap_ngrams(g, t) - 2 / 7) < 1e-12    # 14 truth words -> 7 eight-grams; 2 reproduced

    ap = argparse.ArgumentParser(description="Verbatim conclusion-recall probe.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--limit", type=int, help="first N questions only")
    args = ap.parse_args()

    records = load_pqal()
    dev, test = load_split(records)
    qids = (dev if args.split == "dev" else test)[: args.limit]
    tok, model = load_model(args.model)
    print(f"wrote {run_probe(tok, model, args.model, records, qids, args.split, args.limit).relative_to(ROOT)}")
