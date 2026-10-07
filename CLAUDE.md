# Project: Does Retrieval Make Medical QA More Reliable?

A controlled evaluation of Llama 3 with and without RAG on PubMedQA.

This is a graded university course project whose final deliverable is a
published research paper. Three-member team. The contribution is an
**evaluation study, not a new architecture** — the novelty is in what is
measured and controlled, not in the system built.

---

## 1. Research question

Does retrieval augmentation make an open-weight 8B model more reliable on
medical question answering, and not just more accurate?

Three sub-questions, each of which can fail:

- **RQ1 — What does retrieval buy?** How do accuracy and the confident-error
  rate change from closed-book (C1) to RAG (C2)?
  *H1: RAG raises accuracy over closed-book.*
- **RQ2 — What does retrieval failure cost?** How much of the gap to the
  gold-context ceiling (C3) comes from questions where the source abstract was
  not retrieved?
  *H2: On retrieval misses, RAG's confident-error rate is no lower than
  closed-book.*
- **RQ3 — Does the model know when it doesn't know?** Does confidence fall when
  the key evidence is withheld (C2b)?
  *H3: Confidence is lower in C2b than on C2 retrieval hits. If not, confidence
  is an unsafe abstention signal.*

A null or negative result is a reportable finding. Published work already shows
retrieval degrading performance for 8B models, so "retrieval did not help" is an
acceptable outcome and must not be engineered away.

---

## 2. Experimental conditions

| ID  | Context given to the model | What it isolates |
|-----|----------------------------|------------------|
| C1  | Question only | What the model already knows |
| C2  | Question + top-k retrieved abstracts (source may or may not be retrieved) | End-to-end system behaviour |
| C2b | Same retrieval with the source article removed (retrieve k+1, drop the source) | Behaviour when key evidence is absent |
| C3  | Question + the source abstract, conclusion removed | Ceiling: perfect retrieval |

**Held fixed across every condition:** the questions, model weights, prompt
template (only the context block changes), greedy decoding, and the label parser.

**The measured gaps:**
- `C2 − C1` = what retrieval buys
- `C3 − C2` = what retrieval failure costs
- `C2b vs C2` = does confidence track the evidence?

---

## 3. Dataset — PubMedQA (Jin et al., EMNLP-IJCNLP 2019)

| Subset | Size | Labels | Our use |
|--------|------|--------|---------|
| PQA-L | 1,000 | expert yes/no/maybe | evaluation: 500 dev / 500 test |
| PQA-U | 61.2k | none | retrieval corpus |
| PQA-A | 211.3k | heuristic yes/no | retrieval corpus |

**PQA-L label distribution:** yes 55.2%, no 33.8%, maybe 11.0%.

**Split (decided, frozen):** our test set is the **official PubMedQA test set** —
the 500 PMIDs in `data/test_ground_truth.json` of the PubMedQA GitHub repo; the
remaining 500 are our dev set. Both halves are exactly 276 yes / 169 no / 55 maybe.
Frozen in `data/split.json`. Chosen over a seeded random split so test numbers are
on the same questions as the reference points below and MIRAGE's PubMedQA*.

**Reference points on PQA-L:** majority class 55.2% · best model in the original
paper 68.1% · single human without the conclusion 78.0%.

### Field semantics — this trips people up

- `question` — the article's **title**, phrased as a question
- `context.contexts` — the abstract **without its conclusion**
- `long_answer` — the abstract's **conclusion**, the text that settles the answer
- `final_decision` — the expert label (yes/no/maybe)

So `context` is already conclusion-stripped and is exactly the C3 gold context —
no extra processing needed. `long_answer` must be retained per record because
`corpus.py` needs to know what to strip, and the memorisation probe needs it.

HuggingFace: `load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")`.
Offline fallback: `ori_pqal.json` from the PubMedQA GitHub repo.

---

## 4. Three leakage controls — non-negotiable

1. **Answer leak.** The source abstract's conclusion restates the answer. It is
   removed from that document in the corpus, matching C3. **Copies** of a source
   under another PMID (conclusion intact) are removed too — they would also
   survive the source's removal in C2b. A copy shares ≥50% (and ≥2) of the
   source's long sentences; 2 found, both of 18019905. Largest overlap kept: 12.5%.
2. **Title leak.** Each question *is* its article's title. All documents are
   indexed **without titles**.
3. **Tuning leak.** Prompt, k and thresholds are fixed on the 500 dev questions;
   the 500 test questions are run **once**.

Without these, the RAG numbers measure the benchmark's construction rather than
retrieval, and the study is void.

---

## 5. Module build order

| Stage | Module | Purpose | GPU |
|-------|--------|---------|-----|
| 1 | `data.py` | Load PQA-L, freeze the official 500/500 dev/test split, majority baseline | no |
| 2 | `corpus.py` | Dedupe PQA-L+U+A by PMID and exact text, strip source conclusions, remove source copies, drop titles | no |
| 3 | `index.py` | BM25 first (CPU), then MedCPT + FAISS | MedCPT only |
| 4 | `retrieve.py` | Context builder for C1/C2/C2b/C3; Recall@k of the source | no |
| 5 | `generate.py` | Llama inference, label + label-probability extraction | yes |
| 6 | `evaluate.py` | All metrics, McNemar tests, bootstrap CIs, flip analysis | no |

`metrics.py` is shared by stages 1 and 6.

**Build order is strict: data → corpus → BM25 → generation → evaluation.**
Each stage is tested end-to-end on a ~20-question slice before scaling.

---

## 6. Configuration (initial; to be confirmed on dev)

| Component | Choice | Why |
|-----------|--------|-----|
| Generator | Llama-3.1-8B-Instruct, FP16 (see Hardware), zero-shot | Open weights, fits in 2×16 GB, 128K context |
| Decoding | Greedy (temperature 0); output restricted to one label | Deterministic, repeatable |
| Confidence | Probability of yes/no/maybe at the answer token, renormalised | Needed for risk–coverage and calibration |
| Corpus | PQA-L + PQA-U + PQA-A abstracts, deduped by PMID and exact text, source copies removed: **273,441** docs, no titles, no section headers (`data/corpus_stats.json`) | Document-level leakage control |
| Retrieval unit | One abstract = one document, **no chunking** | Keeps "was the source retrieved?" well defined |
| Sparse retriever | BM25 | Strong lexical baseline, CPU only |
| Dense retriever | MedCPT query + article encoders, exact FAISS inner-product | Biomedical retriever; 768-d index ≈ 0.84 GB |
| Top-k | k = 5 default; sweep k ∈ {1, 3, 5, 10} on dev | Position and dilution effects |
| Hardware | Kaggle 2×T4 (free, 30 GPU-h/week). T4 has no BF16, so FP16 with the model split across both GPUs — **reported as a deviation**. 4-bit only as a reported fallback | Precision is identical in every condition, so condition comparisons stay valid; a small BF16 spot-check on dev quantifies the effect if a ≥24 GB GPU becomes available |

MedCPT's article encoder caps input at 512 tokens — the share of truncated
abstracts must be measured and reported.

---

## 7. Evaluation — accuracy is not enough

**Correctness:** accuracy · macro-F1 · per-class recall, especially `maybe` ·
floor = majority class 55.2%.

**Reliability:** confident-error rate (wrong answer with confidence ≥ τ) ·
flips between C1 and C2 (correct→wrong, wrong→correct) · risk–coverage curve and
AURC · expected calibration error (ECE).

**Retrieval:** Recall@k of the source abstract · RAG results split into
retrieval hits vs misses · accuracy and confidence across k.

**Statistics:** paired McNemar tests between conditions, 95% bootstrap CIs. With
500 test questions, paired differences below roughly 4–6 points may not be
detectable — report CIs, do not claim small differences.

### Two definitions that must stay straight

- **"Hallucination" is operational.** PubMedQA answers are labels, so no free
  text is fabricated and we do not claim to detect fabrication. We measure
  **confident errors**: a committed answer that is wrong with confidence ≥ τ,
  where τ is fixed on dev. In C2 we additionally know whether the source
  abstract was retrieved, so a confident error on a retrieval miss is an answer
  unsupported by what the model was shown.
- **"maybe" is NOT abstention.** It is a gold label (11%) meaning the underlying
  study was inconclusive. Answering `maybe` to a `yes` question is an **error**,
  not a refusal. Abstention is measured separately, by withholding answers whose
  confidence falls below a threshold.

---

## 8. Hard rules

- The dev/test split is frozen in `data/split.json` (official PubMedQA test set,
  see §3). **Changing it mid-project invalidates every number collected before
  the change.**
- Prompt, top-k and τ are tuned on **dev only**.
- The test split is run **once**, at the end.
- Majority baseline must reproduce **55.20% accuracy / 0.2371 macro-F1**. If it
  does not, the data did not load correctly — stop and fix that first.
- No fine-tuning. The study isolates the effect of context, so weights stay
  fixed across all conditions.
- Metrics are hand-implemented, not imported from sklearn, so every number can
  be defended line by line in a viva.

---

## 9. Working style — follow this

- Plain `.py` files in VSCode. **No Jupyter notebooks.**
- **Incremental construction with testing at each stage.** Do not generate large
  amounts of code at once. Previous bulk AI-generated code that was never tested
  end-to-end caused real problems on this project.
- After writing any module, **run it** and show real output before moving on.
- Explain concepts as we go. Every line must be understandable well enough to
  defend in a viva or an interview. Prefer a short explanation of *why* a choice
  was made over a comment restating *what* the line does.
- For anything expensive (the ~273.5k corpus build, indexing, generation), add a
  `--limit` flag so a small slice can be sanity-checked first.
- Be **brutally honest** over encouraging. If something is weak, generic, or
  overclaimed, say so directly. The evaluator asks many follow-up questions and
  every claim has to be defensible.
- Fail loudly: assert on bad data rather than silently producing wrong output.

---

## 10. What this project does NOT claim

- A new RAG architecture
- Clinical safety or readiness for deployment
- Results beyond PubMedQA and the Llama 3 family

Overclaiming novelty is the fastest way to lose the paper in review. The honest
framing: closed-book-vs-RAG comparisons, gold-context evaluation, and abstention
research all already exist separately. What has not been combined is all four
conditions on PubMedQA with a single open 8B model, measuring retrieval success
per question and reporting calibration alongside correctness.

---

## 11. Related-work anchors (for the paper)

- **PubMedQA** — Jin et al., EMNLP-IJCNLP 2019
- **MIRAGE / MedRAG** — Xiong et al., Findings of ACL 2024. Closest accuracy
  study; originates the PubMedQA* context-removed protocol we build on. Cite it
  explicitly as the source of that setting.
- **MedRGB** — Ngo et al., arXiv 2024. Closest abstention study, but uses
  constructed contexts and prompted abstention; we use a real retriever and
  measured confidence.
- **MedCPT** — Jin et al., Bioinformatics 2023. Our dense retriever.
- **Lost in the Middle** — Liu et al., TACL 2024. Motivates the k sweep.
- **When Retrieval Hurts** — ICAISNS 2025. Direct precedent for the flip
  analysis; RAG variants fell below no-retrieval on MedQA.

Recurring gap across the implementation literature: nobody measures whether the
correct source document was actually retrieved, and nobody reports calibration
or a confident-error rate. Retrieval is assumed to work and confidence is
assumed to be meaningful. This project tests both assumptions.
