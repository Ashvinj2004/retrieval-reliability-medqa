"""Turn evaluation results into the paper's tables (Markdown + LaTeX) and figures.

Every number in the paper should come from here, so nothing is copied by hand.
Reads the JSON files written by evaluate.py, the frozen settings, and the
retrieval rankings.

Run:  python report.py --model Llama-3.1-8B-Instruct --tag test     (final paper numbers)
      python report.py --model Llama-3.1-8B-Instruct --tag dev      (dev analysis)
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")  # no screen needed (Kaggle, servers)
import matplotlib.pyplot as plt  # noqa: E402

from data import ROOT  # noqa: E402
from evaluate import KS, SETTINGS  # noqa: E402

RETRIEVERS = ("bm25", "medcpt")
NAMES = {"bm25": "BM25", "medcpt": "MedCPT"}


def pct(x):
    return f"{100 * x:.1f}"


def ci(d):
    return f"{100 * d['diff']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]"


def recall_table(tag):
    """Recall@k of the source per retriever, from the saved rankings."""
    split, *rest = tag.split("_")
    out = {}
    for r in RETRIEVERS:
        path = ROOT / "artifacts" / "retrieval" / f"rankings_{'_'.join([split, r] + rest)}.jsonl"
        if path.exists():
            ranks = [json.loads(line)["source_rank"] for line in open(path, encoding="utf-8")]
            out[r] = {k: sum(1 for x in ranks if x is not None and x <= k) / len(ranks) for k in KS + (100,)}
    return out


def main(model, prompt_version, tag, ks):
    res = {}
    for r in RETRIEVERS:
        if r in ks:
            path = ROOT / "artifacts" / "results" / f"{model}_{prompt_version}_{tag}_{r}_k{ks[r]}.json"
            if path.exists():
                res[r] = json.loads(path.read_text(encoding="utf-8"))
    assert res, f"no evaluate.py results for {model} / {tag}"
    out_dir = ROOT / "artifacts" / "report" / f"{model}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md, tex = [f"# Results: {model}, prompt {prompt_version}, {tag}\n"], []

    # Table 1: every condition. C1 and C3 do not depend on the retriever, so they come from the first one.
    first = next(iter(res.values()))
    rows = [("C1 closed-book", first["conditions"]["C1"]), ("C3 gold context", first["conditions"]["C3"])]
    for r, d in res.items():
        rows += [(f"C2 {NAMES[r]} k={ks[r]}", d["conditions"]["C2"]), (f"C2b {NAMES[r]} k={ks[r]}", d["conditions"]["C2b"])]
    md.append("## Table 1. Correctness and reliability per condition\n")
    md.append("| Condition | n | Acc % | Macro-F1 | Recall(maybe) % | Conf. err % (tau=0.9) | ECE | AURC | Mean conf. | Label mass |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    tex.append("\\begin{tabular}{lrrrrrrrr}\n\\toprule\nCondition & Acc & Macro-F1 & R(maybe) & Conf.\\,err & ECE & AURC & Conf. & Mass \\\\\n\\midrule")
    for name, s in rows:
        md.append(f"| {name} | {s['n']} | {pct(s['accuracy'])} | {s['macro_f1']:.3f} | {pct(s['recall_maybe'])} | "
                  f"{pct(s['confident_error'])} | {s['ece']:.3f} | {s['aurc']:.3f} | {s['mean_conf']:.3f} | {s['mean_label_mass']:.3f} |")
        tex.append(f"{name} & {pct(s['accuracy'])} & {s['macro_f1']:.3f} & {pct(s['recall_maybe'])} & "
                   f"{pct(s['confident_error'])} & {s['ece']:.3f} & {s['aurc']:.3f} & {s['mean_conf']:.3f} & {s['mean_label_mass']:.3f} \\\\")
    tex.append("\\bottomrule\n\\end{tabular}")

    # Table 2: paired comparisons (RQ1, RQ2) and the confidence test (RQ3).
    md.append("\n## Table 2. Paired comparisons (accuracy difference, points, 95% bootstrap CI)\n")
    md.append("| Retriever | C2 - C1 | McNemar p | flips R->W / W->R | C3 - C2 | McNemar p | RQ3: conf. C2b - C2 on hits [CI] (n) |")
    md.append("|---|---|---|---|---|---|---|")
    for r, d in res.items():
        a, b, h = d["C2_vs_C1"], d["C3_vs_C2"], d.get("H3_conf_C2b_minus_C2_on_hits")
        h3 = f"{h['diff']:+.3f} [{h['ci'][0]:+.3f}, {h['ci'][1]:+.3f}] ({h['n']})" if h else "n/a"
        md.append(f"| {NAMES[r]} k={ks[r]} | {ci(a)} | {a['mcnemar_p']:.3f} | {a['right_to_wrong']} / {a['wrong_to_right']} | "
                  f"{ci(b)} | {b['mcnemar_p']:.3f} | {h3} |")

    # Table 3: RQ2 hits vs misses.
    md.append("\n## Table 3. Retrieval hits vs misses in C2 (RQ2 / H2)\n")
    md.append("| Retriever | Group | n | C2 acc % | C2 conf. err % | C1 acc % (same qs) | C1 conf. err % (same qs) |")
    md.append("|---|---|---|---|---|---|---|")
    for r, d in res.items():
        for g in ("hits", "misses"):
            if f"C2_{g}" in d:
                s2, s1 = d[f"C2_{g}"], d[f"C1_on_C2_{g}"]
                md.append(f"| {NAMES[r]} | {g} | {s2['n']} | {pct(s2['accuracy'])} | {pct(s2['confident_error'])} | "
                          f"{pct(s1['accuracy'])} | {pct(s1['confident_error'])} |")

    # Table 4: threshold sensitivity; Table 5: retrieval; Table 6: memorisation.
    md.append("\n## Table 4. Confident-error rate by threshold (sensitivity)\n")
    md.append("| Condition | tau=0.8 | tau=0.9 | tau=0.95 |\n|---|---|---|---|")
    for r, d in res.items():
        for c in ("C1", "C2", "C2b", "C3"):
            if c in ("C1", "C3") and r != next(iter(res)):
                continue
            t = d["tau_sensitivity"][c]
            label = c if c in ("C1", "C3") else f"{c} {NAMES[r]}"
            md.append(f"| {label} | " + " | ".join(pct(t[str(x)] if str(x) in t else t[x]) for x in (0.8, 0.9, 0.95)) + " |")
    rec = recall_table(tag)
    if rec:
        md.append("\n## Table 5. Recall@k of the source abstract\n")
        md.append("| Retriever | " + " | ".join(f"@{k}" for k in KS + (100,)) + " |\n|---|" + "---|" * (len(KS) + 1))
        for r, v in rec.items():
            md.append(f"| {NAMES[r]} | " + " | ".join(pct(v[k]) for k in KS + (100,)) + " |")
    mem = first.get("memorisation")
    if mem:
        md.append("\n## Table 6. Memorisation probe (verbatim conclusion recall)\n")
        md.append(f"- Questions reproduced verbatim (>= 8 leading words): {mem['n_memorised']} of {mem['n_memorised'] + mem['n_clean']}")
        for key, v in mem.items():
            if key.startswith("C1_accuracy"):
                md.append(f"- {key.replace('_', ' ')}: {pct(v)}%")
        for r, d in res.items():
            if "C2_vs_C1_without_memorised" in d.get("memorisation", {}):
                md.append(f"- C2 - C1 without memorised questions, {NAMES[r]}: {ci(d['memorisation']['C2_vs_C1_without_memorised'])}")

    # Table 7: precision / recall / F1 per class (C1, C3, and C2 per retriever).
    if "per_class" in first["conditions"]["C1"]:
        md.append("\n## Table 7. Precision / recall / F1 per class\n")
        md.append("| Condition | yes P | yes R | yes F1 | no P | no R | no F1 | maybe P | maybe R | maybe F1 |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        pc_rows = [("C1", first["conditions"]["C1"]), ("C3", first["conditions"]["C3"])]
        pc_rows += [(f"C2 {NAMES[r]}", d["conditions"]["C2"]) for r, d in res.items()]
        for name, s in pc_rows:
            cells = [f"{100 * s['per_class'][c][m]:.1f}" if m != "f1" else f"{s['per_class'][c][m]:.3f}"
                     for c in ("yes", "no", "maybe") for m in ("precision", "recall", "f1")]
            md.append(f"| {name} | " + " | ".join(cells) + " |")

    # Table 8 (POST HOC, not pre-registered): yes-vs-no judgement on the yes/no questions.
    if "posthoc_binary" in first:
        pb0 = first["posthoc_binary"]
        md.append(f"\n## Table 8. POST HOC: yes-vs-no accuracy on the {pb0['n']} yes/no questions "
                  f"(always 'yes' = {pct(pb0['always_yes'])}%)\n")
        md.append("| Condition | answered 'maybe' % (all qs) | yes-vs-no accuracy % |\n|---|---|---|")
        brow = [("C1", "C1", first), ("C3", "C3", first)] + \
               [(f"{c} {NAMES[r]}", c, d) for r, d in res.items() for c in ("C2", "C2b")]
        for name, c, d in brow:
            pb = d["posthoc_binary"]
            md.append(f"| {name} | {pct(pb['predicted_maybe'][c])} | {pct(pb['accuracy'][c])} |")
        md.append("\n| Retriever | C2 - C1 | C3 - C2 | C2b - C2 |\n|---|---|---|---|")
        for r, d in res.items():
            pb = d["posthoc_binary"]
            md.append(f"| {NAMES[r]} | " + " | ".join(f"{ci(pb[key])} (p={pb[key]['mcnemar_p']:.2g})"
                                                     for key in ("C2_vs_C1", "C3_vs_C2", "C2b_vs_C2")) + " |")

    def save(fig, name):
        # Vector PDF for the paper (sharp at any zoom) plus a 300-dpi PNG for quick viewing.
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.pdf")
        fig.savefig(out_dir / f"{name}.png", dpi=300)
        plt.close(fig)

    # Figure 3: Recall@k of the source per retriever.
    if rec:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for r, v in rec.items():
            ax.plot(list(KS), [100 * v[k] for k in KS], "o-", label=NAMES[r])
        ax.set(xlabel="k (abstracts retrieved)", ylabel="Recall@k of the source (%)", xticks=list(KS))
        ax.grid(alpha=0.3)
        ax.legend()
        save(fig, "fig_recall")

    # Figure 1: risk-coverage curves, one panel per retriever.
    fig, axes = plt.subplots(1, len(res), figsize=(5.2 * len(res), 3.8), squeeze=False)
    for ax, (r, d) in zip(axes[0], res.items()):
        for c, style in (("C1", ":"), ("C2", "-"), ("C2b", "--"), ("C3", "-.")):
            risks = d["risk_curves"][c]
            ax.plot([(i + 1) / len(risks) for i in range(len(risks))], risks, style, label=c)
        ax.set(title=f"{NAMES[r]} (k={ks[r]})", xlabel="coverage (share of questions answered)", ylabel="risk (error rate)")
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, "fig_risk_coverage")

    # Figure 2: dev k-sweep (the pre-registered choice of k).
    if SETTINGS.exists():
        sweep = json.loads(SETTINGS.read_text(encoding="utf-8")).get("dev_c2_macro_f1", {})
        if sweep:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            for r, v in sweep.items():
                ax.plot([int(k) for k in v], list(v.values()), "o-", label=NAMES[r])
            ax.set(xlabel="k (abstracts in context)", ylabel="dev C2 macro-F1", xticks=list(KS))
            ax.grid(alpha=0.3)
            ax.legend()
            save(fig, "fig_k_sweep")

    (out_dir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "table1.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {out_dir.relative_to(ROOT)}/ (results.md, table1.tex, figures as .pdf and .png)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the paper's tables and figures from evaluate.py output.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--tag", default="test")
    ap.add_argument("--k", nargs="*", default=[], help="override, e.g. bm25=5 medcpt=3 (default: frozen settings)")
    args = ap.parse_args()
    ks = json.loads(SETTINGS.read_text(encoding="utf-8"))["k"] if SETTINGS.exists() else {}
    ks.update({kv.split("=")[0]: int(kv.split("=")[1]) for kv in args.k})
    main(args.model, args.prompt_version, args.tag, ks)
