# Run the experiments (about 15 minutes of your time, then ~7-9 hours unattended)

Everything else is automated by `run_all.py`: it rebuilds and checks the data,
runs both retrievers, asks the model every question, applies the pre-registered
rules to choose settings on the dev split, runs the test split once, and builds
the paper's tables and figures.

## 1. Kaggle account (5 min)

1. Sign in at **kaggle.com**.
2. **Settings → Phone verification.** Required, otherwise GPUs and internet stay off.

## 2. Optional: Llama instead of Qwen (skip if short on time)

The script uses **Llama-3.1-8B-Instruct** if it finds a HuggingFace token with
Llama access, and otherwise **Qwen2.5-7B-Instruct** (same size class, no approval
needed). To use Llama:

1. At **huggingface.co/meta-llama/Llama-3.1-8B-Instruct**, request access and wait for approval.
2. HuggingFace **Settings → Access Tokens → New token** (type *Read*).
3. In the Kaggle notebook (step 3): **Add-ons → Secrets → Add secret**, label
   `HF_TOKEN`, paste the token, switch it **on**.

## 3. The notebook (5 min)

1. **Create → New Notebook.**
2. Right panel → **Session options**: **Accelerator: GPU T4 x2**, **Internet: On**.
3. Delete the sample cell and paste this single cell:

```
!git clone -b finish https://github.com/Ashvinj2004/retrieval-reliability-medqa.git && cd retrieval-reliability-medqa && python run_all.py
```

4. Top right: **Save Version → Save & Run All (Commit) → Save.**
   It now runs in the background; you can close the browser.

## 4. Collect the result (2 min, 7-9 hours later)

1. Open the notebook → **Versions** → the finished version → **Output** tab.
2. Download **`retrieval-reliability-medqa/results.zip`**.
3. Put it in the project folder (`C:\Projects\DL RAG Project\`) and say "results are in".

## If the run fails

Open the version's **Log**. Every step prints `$ python <script>`; the last one
shown is the one that failed, and the scripts say why (they stop on purpose
when something is wrong). Send the last ~30 lines of the log.

| Message | Meaning |
|---|---|
| `rebuilt corpus differs from the committed one` | the data changed upstream; stop and report |
| `format check failed` | the model did not answer in one word; the prompt needs revising on dev |
| `CUDA out of memory` | the accelerator is not **GPU T4 x2** |
| `Could not resolve host` / pip errors | Internet is off, or phone verification is missing |
