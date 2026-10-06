"""Hand-implemented evaluation metrics, shared by data.py (stage 1) and evaluate.py (stage 6).

Written by hand rather than imported from sklearn (CLAUDE.md section 8) so every
reported number can be traced line by line. This file starts with the correctness
metrics; calibration, risk-coverage, McNemar and bootstrap are added in stage 6.

Run `python metrics.py` to execute the self-test.
"""

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

    print("metrics.py self-test passed")
