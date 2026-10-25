"""Stage 5: ask the model, read its probability for yes / no / maybe.

One forward pass per question, no free text: at the answer position we read the
model's probability for each label word (case variants such as "Yes"/"yes" are
summed), renormalise the three to sum to 1, and take the largest. That is exactly
greedy decoding restricted to the three labels, and it gives the confidence for
free. We also keep `label_mass`, the probability the three labels held BEFORE
renormalising: if it is low, the model wanted to say something else first.

Host-independent: reads context files written by retrieve.py, writes one JSON
line per question. The prompt template is identical in every condition; only
the documents block changes (and is absent in C1).

Run:  python generate.py --contexts artifacts/retrieval/contexts_dev_q20_C3.jsonl --dry-run
      python generate.py --contexts ... --model HuggingFaceTB/SmolLM2-135M-Instruct   (CPU pipeline test)
      python generate.py --contexts ...                                              (Llama, on Kaggle)
"""
import argparse
import json
import time
from pathlib import Path

from data import ROOT, load_pqal
from metrics import LABELS

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_VERSION = "v1"  # bump on ANY wording change; tuned on dev only (CLAUDE.md section 8)
TEMPLATE_DATE = "26 Jul 2024"  # Llama 3.1's own template default, fixed so the prompt never depends on the run date

SYSTEM = (
    "You are a biomedical research assistant. Answer the research question with exactly one word: "
    "yes, no, or maybe. Use maybe only when the evidence is inconclusive."
)


def build_messages(question, documents):
    """Chat messages for one question. Only the documents block differs between conditions."""
    parts = []
    if documents:
        parts.append("Abstracts:\n" + "\n\n".join(f"[{i}] {d}" for i, d in enumerate(documents, 1)))
    parts.append(f"Question: {question}\nAnswer (yes, no, or maybe):")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def label_token_ids(tokenizer):
    """Token ids for each label and its case variants. Each variant must be ONE token,
    otherwise its first-token probability would not be the probability of the word."""
    ids = {}
    for label in LABELS:
        variants = {label, label.capitalize()}
        ids[label] = []
        for v in sorted(variants):
            toks = tokenizer.encode(v, add_special_tokens=False)
            assert len(toks) == 1, f"{v!r} is {len(toks)} tokens for this tokenizer: {toks}"
            ids[label].append(toks[0])
    flat = [t for ts in ids.values() for t in ts]
    assert len(flat) == len(set(flat)), "two labels share a token id"
    return ids


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if torch.cuda.is_available():
        # FP16 on Kaggle's T4s (no BF16); device_map splits the 8B model over both GPUs.
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    return tok, model.eval()


def score(tok, model, label_ids, messages):
    """Renormalised label probabilities, label mass, and prompt length for one question."""
    import torch

    # Llama 3.1's template writes a "Today Date" line into the system prompt; pinning it keeps the
    # prompt identical whatever day the run happens (templates without that variable ignore it).
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, date_string=TEMPLATE_DATE)
    ids = enc["input_ids"]
    with torch.no_grad():
        logits = model(ids.to(model.device)).logits[0, -1].float()  # next-token logits at the answer position
    p = torch.softmax(logits, dim=-1)
    raw = {lab: float(p[t].sum()) for lab, t in label_ids.items()}
    mass = sum(raw.values())
    return {lab: v / mass for lab, v in raw.items()}, mass, int(ids.shape[1])


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score yes/no/maybe probabilities for one context file.")
    ap.add_argument("--contexts", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, help="first N questions only")
    ap.add_argument("--dry-run", action="store_true", help="print the first prompt and stop; no model needed")
    args = ap.parse_args()

    records = load_pqal()
    rows = read_jsonl(args.contexts)[: args.limit]
    assert all("label" not in r for r in rows), "context files must not carry gold labels"

    if args.dry_run:
        r = rows[0]
        for m in build_messages(records[r["qid"]]["question"], r["documents"]):
            print(f"--- {m['role']} ---\n{m['content'][:1500]}{' [...]' if len(m['content']) > 1500 else ''}")
        raise SystemExit(0)

    tok, model = load_model(args.model)
    label_ids = label_token_ids(tok)
    out_path = ROOT / "artifacts" / "generations" / f"{args.model.split('/')[-1]}_{PROMPT_VERSION}_{args.contexts.stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "wb") as f:
        for i, r in enumerate(rows, 1):
            probs, mass, n_tok = score(tok, model, label_ids, build_messages(records[r["qid"]]["question"], r["documents"]))
            pred = max(LABELS, key=probs.get)
            row = {"qid": r["qid"], "condition": r["condition"], "retriever": r["retriever"], "k": r["k"],
                   "source_in_context": r["source_in_context"], "pred": pred, "confidence": probs[pred],
                   "probs": probs, "label_mass": mass, "prompt_tokens": n_tok,
                   "model": args.model, "prompt_version": PROMPT_VERSION}
            f.write((json.dumps(row) + "\n").encode("utf-8"))
            if i % 50 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} ({(time.time() - t0) / i:.2f}s per question)")
    print(f"wrote {out_path.relative_to(ROOT)}")
