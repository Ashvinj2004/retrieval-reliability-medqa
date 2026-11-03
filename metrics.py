"""Hand-implemented evaluation metrics, shared by data.py (stage 1) and evaluate.py (stage 6).

Written by hand rather than imported from sklearn (CLAUDE.md section 8) so every
reported number can be traced line by line. This file starts with the correctness
metrics; calibration, risk-coverage, McNemar and bootstrap are added in stage 6.

Run `python metrics.py` to execute the self-test.
"""

import math

import numpy as np

# Fixed label set and order. Fixing it here, rather than reading labels from the
# data, means a class the model never predicts still counts (as F1 = 0) in macro-F1.
LABELS = ("yes", "no", "maybe")


def _check(gold, pred):
    """Fail loudly on mismatched lengths or labels outside LABELS (e.g. 'Yes', None)."""
    assert len(gold) == len(pred) > 0, f"need equal, non-zero lengths (gold={len(gold)}, pred={len(pred)})"
    bad = {x for x in (*gold, *pred) if x not in LABELS}
    assert not bad, f"labels outside {LABELS}: {bad}"


def accuracy(gold, pred):
    _check(gold, pred)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def confusion_matrix(gold, pred):
    """cm[g][p] = number of questions with gold label g that were predicted as p."""
    _check(gold, pred)
    cm = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        cm[g][p] += 1
    return cm


def per_class(gold, pred):
    """Precision, recall, F1 and support (number of gold examples) for each label.

    precision(c) = TP / times c was predicted      (a column of the confusion matrix)
    recall(c)    = TP / times c was the gold label (a row of the confusion matrix)
    A ratio with a zero denominator (e.g. a label that is never predicted) is
    defined as 0, the usual convention. This is why always answering 'yes'
    scores F1 = 0 on 'no' and 'maybe'.
    """
    cm = confusion_matrix(gold, pred)
    out = {}
    for c in LABELS:
        tp = cm[c][c]
        n_pred = sum(cm[g][c] for g in LABELS)
        n_gold = sum(cm[c][p] for p in LABELS)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1, "support": n_gold}
    return out


def macro_f1(gold, pred):
    """Unweighted mean of the three per-class F1 scores.

    Unlike accuracy, 'maybe' (11% of questions) counts as much as 'yes' (55%),
    so a model that ignores the rare class cannot hide behind the common one.
    """
    pc = per_class(gold, pred)
    return sum(pc[c]["f1"] for c in LABELS) / len(LABELS)


# ---------------------------------------------------------------- reliability (stage 6)

def confident_error_rate(gold, pred, conf, tau):
    """Share of ALL questions answered wrongly with confidence >= tau (CLAUDE.md's operational 'hallucination')."""
    _check(gold, pred)
    assert len(conf) == len(gold) and all(0.0 <= c <= 1.0 for c in conf), "confidences must be in [0, 1]"
    return sum(g != p and c >= tau for g, p, c in zip(gold, pred, conf)) / len(gold)


def ece(gold, pred, conf, n_bins=10):
    """Expected calibration error with equal-width confidence bins.

    ECE = sum over bins of (share of questions in the bin) * |accuracy in bin - mean confidence in bin|.
    0 means that answers given with 80% confidence are right 80% of the time, and so on.
    """
    _check(gold, pred)
    bins = [[] for _ in range(n_bins)]
    for g, p, c in zip(gold, pred, conf):
        bins[min(int(c * n_bins), n_bins - 1)].append((g == p, c))  # confidence 1.0 goes in the top bin
    total = 0.0
    for b in bins:
        if b:
            acc = sum(ok for ok, _ in b) / len(b)
            mean_conf = sum(c for _, c in b) / len(b)
            total += len(b) / len(gold) * abs(acc - mean_conf)
    return total


def risk_coverage(gold, pred, conf):
    """Answer the most confident questions first; after i answers, risk = errors so far / i.

    Returns (coverages, risks) and the AURC (mean risk over all i). Low AURC means
    confidence ranks right answers above wrong ones, i.e. it is a usable abstention signal.
    Ties in confidence are broken by input order, so the result is deterministic.
    """
    _check(gold, pred)
    order = sorted(range(len(gold)), key=lambda i: -conf[i])
    errors, coverages, risks = 0, [], []
    for n, i in enumerate(order, 1):
        errors += gold[i] != pred[i]
        coverages.append(n / len(gold))
        risks.append(errors / n)
    return coverages, risks, sum(risks) / len(risks)


# ---------------------------------------------------------------- statistics (stage 6)

def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test for two conditions answered on the SAME questions.

    Only discordant questions carry information: b = A right & B wrong, c = A wrong & B right.
    If the conditions are equally good, each discordant question is a fair coin flip, so
    p = 2 * P(Binomial(b + c, 0.5) <= min(b, c)), capped at 1. Exact, because b + c can be small.
    """
    assert len(correct_a) == len(correct_b) > 0
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def paired_bootstrap_ci(values_a, values_b, n_boot=10_000, seed=0, alpha=0.05):
    """95% percentile CI for mean(values_b) - mean(values_a), resampling QUESTIONS (paired).

    values_* are per-question numbers (e.g. 1/0 correctness) in the same question order.
    The fixed seed makes every reported interval reproducible.
    """
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))  # each row = one resample of the questions
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def flips(correct_a, correct_b):
    """How answers change from condition A to B: (right->wrong, wrong->right)."""
    return (sum(1 for x, y in zip(correct_a, correct_b) if x and not y),
            sum(1 for x, y in zip(correct_a, correct_b) if y and not x))


if __name__ == "__main__":
    # 1) A 4-question example small enough to check with pen and paper.
    gold = ["yes", "yes", "no", "maybe"]
    pred = ["yes", "no", "no", "yes"]
    #   yes:   predicted at q0,q3 (1 right) -> P = 1/2;  gold at q0,q1 (1 found) -> R = 1/2;  F1 = 1/2
    #   no:    predicted at q1,q2 (1 right) -> P = 1/2;  gold at q2    (1 found) -> R = 1;    F1 = 2/3
    #   maybe: never predicted              -> P = 0;    gold at q3    (0 found) -> R = 0;    F1 = 0
    #   accuracy = 2/4;  macro-F1 = (1/2 + 2/3 + 0) / 3 = 7/18
    pc = per_class(gold, pred)
    assert accuracy(gold, pred) == 0.5
    assert abs(pc["yes"]["f1"] - 1 / 2) < 1e-12
    assert abs(pc["no"]["f1"] - 2 / 3) < 1e-12
    assert pc["maybe"]["f1"] == 0.0
    assert abs(macro_f1(gold, pred) - 7 / 18) < 1e-12
    print(f"toy example:  accuracy={accuracy(gold, pred):.4f}  macro-F1={macro_f1(gold, pred):.4f}"
          "  (by hand: 0.5000 / 0.3889)")

    # 2) Majority baseline in closed form: always 'yes' on PQA-L's 552 / 338 / 110.
    #    accuracy = 0.552;  F1(yes) = 2(0.552)(1) / (0.552 + 1) = 0.7113;  macro-F1 = 0.7113 / 3 = 0.2371
    gold = ["yes"] * 552 + ["no"] * 338 + ["maybe"] * 110
    pred = ["yes"] * 1000
    acc, mf1 = accuracy(gold, pred), macro_f1(gold, pred)
    assert abs(acc - 0.5520) < 5e-5 and abs(mf1 - 0.2371) < 5e-5
    print(f"always 'yes': accuracy={acc:.4f}  macro-F1={mf1:.4f}  (paper: 0.5520 / 0.2371)")

    # 3) Bad input must fail loudly, not be silently scored as a wrong answer.
    try:
        accuracy(["yes"], ["Yes"])
    except AssertionError as e:
        print(f"bad label rejected as intended: {e}")
    else:
        raise SystemExit("ERROR: capitalised 'Yes' was accepted")

    # 4) Reliability metrics on a 4-question example.
    gold = ["yes", "yes", "no", "no"]
    pred = ["yes", "no", "no", "yes"]          # right, wrong, right, wrong
    conf = [0.95, 0.95, 0.55, 0.55]
    #   confident errors at tau=0.9: only q1 (wrong, 0.95) -> 1/4
    #   ECE: bin 0.9-1.0 has acc 1/2, conf 0.95 -> 0.45 * 2/4;  bin 0.5-0.6 has acc 1/2, conf 0.55 -> 0.05 * 2/4
    #        ECE = 0.225 + 0.025 = 0.25
    assert confident_error_rate(gold, pred, conf, 0.9) == 0.25
    assert abs(ece(gold, pred, conf) - 0.25) < 1e-12
    #   risk-coverage with confidences 0.9, 0.8, 0.7, 0.6 and correctness 1, 0, 1, 0:
    #   risks 0/1, 1/2, 1/3, 2/4 -> AURC = (0 + 0.5 + 0.3333 + 0.5) / 4 = 1/3
    _, risks, aurc = risk_coverage(gold, ["yes", "no", "no", "yes"], [0.9, 0.8, 0.7, 0.6])
    assert abs(aurc - 1 / 3) < 1e-12
    print(f"reliability:  confident-error={0.25:.4f}  ECE={ece(gold, pred, conf):.4f}  AURC={aurc:.4f}"
          "  (by hand: 0.2500 / 0.2500 / 0.3333)")

    # 5) Exact McNemar: b = 10, c = 2 discordant questions.
    #    p = 2 * (C(12,0) + C(12,1) + C(12,2)) / 2^12 = 2 * 79 / 4096 = 0.03857
    a = [True] * 10 + [False] * 2 + [True] * 30
    b = [False] * 10 + [True] * 2 + [True] * 30
    nb, nc, p = mcnemar_exact(a, b)
    assert (nb, nc) == (10, 2) and abs(p - 2 * 79 / 4096) < 1e-12
    assert flips(a, b) == (10, 2)
    print(f"McNemar:      b={nb} c={nc} p={p:.5f}  (by hand: 0.03857)")

    # 6) Paired bootstrap: identical systems give exactly [0, 0]; the interval is reproducible.
    assert paired_bootstrap_ci(a, a) == (0.0, 0.0, 0.0)
    d, lo, hi = paired_bootstrap_ci(a, b)
    assert lo <= d <= hi and (d, lo, hi) == paired_bootstrap_ci(a, b)
    print(f"bootstrap:    diff={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (same on rerun)")

    print("metrics.py self-test passed")
