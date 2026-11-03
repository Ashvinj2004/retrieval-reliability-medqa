"""Stage 6: turn generation files into the numbers that answer RQ1-RQ3.

All metrics come from metrics.py (hand-written). Every comparison is PAIRED: the
conditions must cover exactly the same questions, which is asserted.

  RQ1  C2 vs C1   accuracy and confident errors; McNemar, bootstrap CI, flips
  RQ2  C2 split into retrieval hits / misses; on misses, C2 vs C1 confident errors (H2); C3 vs C2
  RQ3  confidence in C2b vs C2 on the same retrieval-hit questions (H3)

Run:  python evaluate.py --model SmolLM2-135M-Instruct --tag dev_q20      (pipeline test)
      python evaluate.py --model Llama-3.1-8B-Instruct --tag dev --tau ...  (real results)
"""
import argparse
import json

from data import ROOT, load_pqal
from metrics import (accuracy, confident_error_rate, ece, flips, macro_f1, mcnemar_exact,
                     paired_bootstrap_ci, per_class, risk_coverage)

GEN = ROOT / "artifacts" / "generations"


def load_condition(model, prompt_version, tag, retriever, k, condition):
    """Generation rows for one condition, keyed by question id."""
    split, *rest = tag.split("_")  # e.g. "dev_q20" -> contexts_dev_bm25_q20_k5_C2
    stem = f"contexts_{tag}_{condition}" if condition in ("C1", "C3") else \
        f"contexts_{'_'.join([split, retriever] + rest)}_k{k}_{condition}"
    path = GEN / f"{model}_{prompt_version}_{stem}.jsonl"
    assert path.exists(), f"missing generation file: {path.relative_to(ROOT)}"
    with open(path, encoding="utf-8") as f:
        rows = {r["qid"]: r for r in map(json.loads, f)}
    assert all(r["prompt_version"] == prompt_version for r in rows.values()), f"{path.name}: mixed prompt versions"
    return rows


def summary(rows, qids, gold, tau):
    g = [gold[q] for q in qids]
    p = [rows[q]["pred"] for q in qids]
    c = [rows[q]["confidence"] for q in qids]
    return {
        "n": len(qids), "accuracy": accuracy(g, p), "macro_f1": macro_f1(g, p),
        "recall_maybe": per_class(g, p)["maybe"]["recall"],
        "confident_error": confident_error_rate(g, p, c, tau), "ece": ece(g, p, c),
        "aurc": risk_coverage(g, p, c)[2], "mean_conf": sum(c) / len(c),
        "mean_label_mass": sum(rows[q]["label_mass"] for q in qids) / len(qids),
    }


def compare(name, rows_a, rows_b, qids, gold):
    """Paired accuracy comparison of B against A on the same questions."""
    ca = [rows_a[q]["pred"] == gold[q] for q in qids]
    cb = [rows_b[q]["pred"] == gold[q] for q in qids]
    d, lo, hi = paired_bootstrap_ci(ca, cb)
    b, c, p = mcnemar_exact(ca, cb)
    rw, wr = flips(ca, cb)
    print(f"  {name:<10} accuracy diff {d:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  McNemar p={p:.3f}"
          f"  flips: right->wrong {rw}, wrong->right {wr}")
    return {"diff": d, "ci": [lo, hi], "mcnemar_p": p, "right_to_wrong": rw, "wrong_to_right": wr}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate one model / prompt / retrieval setting.")
    ap.add_argument("--model", required=True, help="model name as in the generation file names")
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--tag", default="dev", help="split tag used by retrieve.py, e.g. dev or dev_q20")
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--tau", type=float, default=0.9, help="confident-error threshold (pre-registered 0.9, RESEARCH_PLAN.md section 6)")
    args = ap.parse_args()
    assert not args.tag.startswith("test"), "test results are produced once, by the final run only"

    gold = {q: r["label"] for q, r in load_pqal().items()}
    cond = {c: load_condition(args.model, args.prompt_version, args.tag, args.retriever, args.k, c)
            for c in ("C1", "C2", "C2b", "C3")}
    qids = sorted(cond["C1"], key=int)
    for c, rows in cond.items():
        assert sorted(rows, key=int) == qids, f"{c} covers different questions: comparisons would not be paired"

    print(f"{args.model} | prompt {args.prompt_version} | {args.tag} | {args.retriever} k={args.k} | tau={args.tau}\n")
    results = {"settings": vars(args), "conditions": {}}
    print(f"  {'cond':<5}{'n':>5}{'acc':>8}{'macroF1':>9}{'R(maybe)':>10}{'conf.err':>10}{'ECE':>7}{'AURC':>7}"
          f"{'conf':>7}{'mass':>7}")
    for c, rows in cond.items():
        s = summary(rows, qids, gold, args.tau)
        results["conditions"][c] = s
        print(f"  {c:<5}{s['n']:>5}{s['accuracy']:>8.1%}{s['macro_f1']:>9.3f}{s['recall_maybe']:>10.1%}"
              f"{s['confident_error']:>10.1%}{s['ece']:>7.3f}{s['aurc']:>7.3f}{s['mean_conf']:>7.3f}"
              f"{s['mean_label_mass']:>7.3f}")

    hits = [q for q in qids if cond["C2"][q]["source_in_context"]]
    misses = [q for q in qids if not cond["C2"][q]["source_in_context"]]
    print(f"\nRQ1 / RQ2 paired comparisons ({len(qids)} questions):")
    results["C2_vs_C1"] = compare("C2 vs C1", cond["C1"], cond["C2"], qids, gold)
    results["C3_vs_C2"] = compare("C3 vs C2", cond["C2"], cond["C3"], qids, gold)

    print(f"\nRQ2 retrieval hits ({len(hits)}) vs misses ({len(misses)}) in C2:")
    for name, group in (("hits", hits), ("misses", misses)):
        if group:
            s2, s1 = summary(cond["C2"], group, gold, args.tau), summary(cond["C1"], group, gold, args.tau)
            results[f"C2_{name}"], results[f"C1_on_C2_{name}"] = s2, s1
            print(f"  {name:<7} C2 acc {s2['accuracy']:.1%} conf.err {s2['confident_error']:.1%}"
                  f"  | C1 on same questions: acc {s1['accuracy']:.1%} conf.err {s1['confident_error']:.1%}")
    if len(misses) < 30:
        print(f"  NOTE: only {len(misses)} misses -- too few for a reliable H2 comparison; report as descriptive.")

    if hits:
        conf_c2 = [cond["C2"][q]["confidence"] for q in hits]
        conf_c2b = [cond["C2b"][q]["confidence"] for q in hits]
        d, lo, hi = paired_bootstrap_ci(conf_c2, conf_c2b)
        results["H3_conf_C2b_minus_C2_on_hits"] = {"diff": d, "ci": [lo, hi], "n": len(hits)}
        print(f"\nRQ3 confidence when the source is withheld (C2b - C2, same {len(hits)} hit questions): "
              f"{d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        print("  H3 predicts a clearly negative difference; a CI including 0 means confidence did not track the evidence.")

    out = ROOT / "artifacts" / "results" / f"{args.model}_{args.prompt_version}_{args.tag}_{args.retriever}_k{args.k}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nsaved {out.relative_to(ROOT)}")
