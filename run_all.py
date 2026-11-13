"""Run the whole study, start to finish, in one command (made for one Kaggle session).

Every decision left was fixed in advance (RESEARCH_PLAN.md section 6), so this
script applies the rules mechanically, without anyone looking at results first:

  0  model: Llama-3.1-8B-Instruct if a HuggingFace token with access is available,
     otherwise Qwen2.5-7B-Instruct (no approval needed; reported as a deviation)
  1  rebuild the corpus; its SHA-256 must equal the committed one
  2  BM25 + MedCPT retrieval for the dev questions
  3  20-question format check: stop if mean label_mass < 0.95
  4  dev answers for C1, C3 and C2 at every k (all the k rule needs)
  5  choose k per retriever by the rule, freeze data/final_settings.json, then C2b at the
     chosen k only (RQ3 uses C2b at that k; other k would cost GPU time for nothing), dev analysis
  6  the test run, ONCE, with the frozen settings
  7  evaluate test, build tables and figures, zip everything into results.zip

Resumable: rerunning skips finished files. Run:
      python run_all.py            (Kaggle, GPU)
      python run_all.py --smoke    (laptop: tiny model, corpus slice, 20 questions)
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import date

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Lets PyTorch reuse fragmented GPU memory (the fix its own out-of-memory message suggests). Set
# before torch is imported, and inherited by every script this one starts.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
SMOKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
MIN_LABEL_MASS = 0.95  # pre-registered format check


def ensure_packages():
    missing = [p for p in ("snowballstemmer", "accelerate", "scipy", "datasets", "matplotlib")
               if importlib.util.find_spec(p) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def run(*args):
    """Run one of our scripts; any failure stops everything (the scripts fail loudly on bad data)."""
    print(f"\n$ python {' '.join(args)}", flush=True)
    assert subprocess.run([sys.executable, *args], cwd=ROOT_DIR).returncode == 0, f"step failed: {' '.join(args)}"


def kaggle_token():
    """HF_TOKEN from the environment or a Kaggle secret, if there is one."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        return token
    except Exception:
        return None


def choose_model():
    if not kaggle_token():
        return QWEN, "no HF_TOKEN available"
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(LLAMA, "config.json")  # raises if the token has no access to the gated repo
        return LLAMA, "HF_TOKEN has Llama access"
    except Exception as e:
        return QWEN, f"Llama not accessible with this token ({type(e).__name__})"


def main(smoke):
    sys.stdout.reconfigure(line_buffering=True)  # progress lines appear in the Kaggle log as they happen
    ensure_packages()
    from data import ROOT, load_pqal, load_split
    from evaluate import KS, SETTINGS
    from generate import load_model, read_jsonl, run_file
    from probe import run_probe
    from retrieve import OUT, tag

    log = {"started": time.strftime("%Y-%m-%d %H:%M"), "smoke": smoke}
    model_name, why = (SMOKE_MODEL, "smoke test") if smoke else choose_model()
    short = model_name.split("/")[-1]
    log.update(model=model_name, model_reason=why)
    print(f"model: {model_name} ({why})", flush=True)
    n_q, lim = (20, 2000) if smoke else (None, None)
    qargs = ["--questions", str(n_q), "--corpus-limit", str(lim)] if smoke else []
    retrievers = ("bm25", "medcpt")

    # 1-2: data, corpus, both retrievers on dev.
    run("data.py")
    run("corpus.py", *(["--limit", str(lim)] if smoke else []))
    if not smoke:
        committed = subprocess.run(["git", "show", "HEAD:data/corpus_stats.json"], cwd=ROOT_DIR,
                                   capture_output=True, text=True, check=True).stdout
        rebuilt = (ROOT / "data" / "corpus_stats.json").read_text(encoding="utf-8")
        assert json.loads(committed)["sha256"] == json.loads(rebuilt)["sha256"], \
            "rebuilt corpus differs from the committed one: results would not match the paper"
    run("index.py", *(["--limit", str(lim)] if smoke else []))
    meta = ROOT / "artifacts" / ("medcpt" if not smoke else f"medcpt_limit{lim}") / "meta.json"
    run("medcpt.py", *(["--limit", str(lim)] if smoke else []), *(["--skip-encode"] if meta.exists() else []),
        "--batch-size", "128")
    for r in retrievers:
        run("retrieve.py", "--retriever", r, *qargs)

    records = load_pqal()
    dev, test = load_split(records)

    def context_files(split, ks_by_retriever, conditions=("C2", "C2b"), with_c1_c3=True):
        base = tag(split, None, n_q, lim)
        files = [OUT / f"contexts_{base}_C1.jsonl", OUT / f"contexts_{base}_C3.jsonl"] if with_c1_c3 else []
        for r, ks in ks_by_retriever.items():
            t = tag(split, r, n_q, lim)
            files += [OUT / f"contexts_{t}_k{k}_{c}.jsonl" for k in ks for c in conditions]
        return files

    def work(path):
        """Rough cost of a context file: one unit per question plus one per document it carries."""
        return sum(1 + len(r["doc_pmids"]) for r in read_jsonl(path))

    t_start = time.time()
    done_units = 0.0

    def generate_all(files, planned_after):
        """Score files and print a MEASURED projection of the total run time (Kaggle stops runs at 12 h)."""
        nonlocal done_units
        for f in files:
            t0 = time.time()
            out = run_file(tok, model, model_name, f, records)
            if time.time() - t0 > 5:  # only files actually computed in this session count toward the rate
                done_units += work(f)
                rate = (time.time() - t_start) / done_units
                remaining = sum(work(x) for x in files[files.index(f) + 1:]) + planned_after
                eta_h = (time.time() - t_start + rate * remaining) / 3600
                print(f"  [time] {(time.time() - t_start) / 3600:.1f} h elapsed; projected total ~{eta_h:.1f} h"
                      f"{'  WARNING: above Kaggle 12 h limit' if eta_h > 11.5 else ''}", flush=True)
            del out

    # 3: format check, then 4: every dev answer. The model is loaded once.
    tok, model = load_model(model_name)
    dev_tag = tag("dev", None, n_q, lim)
    check = run_file(tok, model, model_name, OUT / f"contexts_{dev_tag}_C3.jsonl", records, limit=20)
    mass = sum(r["label_mass"] for r in read_jsonl(check)) / 20
    log["format_check_label_mass"] = mass
    print(f"\nformat check: mean label_mass on 20 dev questions = {mass:.3f} (must be >= {MIN_LABEL_MASS})", flush=True)
    if not smoke:  # the tiny smoke model cannot pass this; only the plumbing is tested there
        assert mass >= MIN_LABEL_MASS, "format check failed: the prompt must be revised on dev before any result counts"
    # The k rule needs C2 at every k; C2b is analysed only at the chosen k, so it waits for the freeze.
    dev_pass1 = context_files("dev", {r: KS for r in retrievers}, conditions=("C2",))
    later = sum(work(f) for f in context_files("dev", {r: [5] for r in retrievers}, ("C2b",), False))  # ~C2b at chosen k
    later += sum(work(f) for f in context_files("dev", {r: [5] for r in retrievers}))  # ~the test run
    generate_all(dev_pass1, planned_after=later)

    # 5: the pre-registered k rule, frozen before any test question is run; dev analysis and probe.
    if SETTINGS.exists() and not json.loads(SETTINGS.read_text(encoding="utf-8")).get("test_started"):
        SETTINGS.unlink()  # a stale freeze from an interrupted run is redone from the same dev files
    if not SETTINGS.exists():
        for r in retrievers:
            run("evaluate.py", "--model", short, "--tag", dev_tag, "--retriever", r, "--sweep", "--freeze")
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    chosen = {r: [settings["k"][r]] for r in retrievers}
    generate_all(context_files("dev", chosen, ("C2b",), False),
                 planned_after=sum(work(f) for f in context_files("dev", chosen)))  # ~the test run
    run_probe(tok, model, model_name, records, dev[:n_q], "dev", n_q)
    for r in retrievers:
        run("evaluate.py", "--model", short, "--tag", dev_tag, "--retriever", r, "--k", str(settings["k"][r]))
    run("report.py", "--model", short, "--tag", dev_tag)

    # 6: the single test run. In smoke mode the same 20 DEV questions stand in for test, so the
    # code path is exercised without ever touching test with a model that is not the real one.
    final_split = "dev" if smoke else "test"
    settings.update(test_started=True, test_started_on=str(date.today()))
    SETTINGS.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    if not smoke:
        for r in retrievers:
            run("retrieve.py", "--retriever", r, "--split", "test")
    generate_all(context_files(final_split, chosen), planned_after=0)
    run_probe(tok, model, model_name, records, (dev if smoke else test)[:n_q], final_split, n_q)

    # 7: final evaluation (k and tau read from the frozen settings), report, one zip.
    final_tag = tag(final_split, None, n_q, lim)
    for r in retrievers:
        run("evaluate.py", "--model", short, "--tag", final_tag, "--retriever", r, "--final")
    run("report.py", "--model", short, "--tag", final_tag)
    log["finished"] = time.strftime("%Y-%m-%d %H:%M")
    (ROOT / "artifacts" / "run_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    zpath = ROOT / ("results_smoke.zip" if smoke else "results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("generations", "probe", "results", "report"):
            for p in (ROOT / "artifacts" / sub).rglob("*"):
                if p.is_file():
                    z.write(p, p.relative_to(ROOT))
        for p in (SETTINGS, ROOT / "artifacts" / "run_log.json", meta, ROOT / "data" / "corpus_stats.json"):
            if p.exists():
                z.write(p, p.relative_to(ROOT))
    print(f"\nDONE. Everything is in {zpath.name} ({zpath.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the whole study in one command.")
    ap.add_argument("--smoke", action="store_true", help="laptop test: tiny model, corpus slice, 20 questions")
    main(ap.parse_args().smoke)
