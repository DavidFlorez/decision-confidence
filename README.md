# Do Decision Model Probabilities Hold Out of Distribution?

![When Decision Models Say They Are 93% Sure — Overview](assets/cover.png)

> **Benchmark claim tested**: *"When Drex Max is at least 93% sure, it is right 97% of the time."* — [Nace AI, Nov 6, 2026 report](https://www.nace.ai/research/towards-foundation-decision-models) (`arXiv:2611.07854`)

Two open-weight non-generative decision models — **[Drex Max v1](https://huggingface.co/nace-ai/drex-max-v1)** (Nace AI, 27B) and **[d1-2B](https://huggingface.co/LiquidAI/d1-2B)** (Liquid AI, 1.9B) — output direct choice probability distributions in a single forward pass instead of generating tokens. Alongside [Artificial Analysis Decision Index v2](https://x.com/ArtificialAnlys/status/2119578471559213411) introducing calibration scoring and Nace announcing open weights for [Drex Max v1.1 "soon"](https://x.com/NaceAI/status/2119792132558819447), we test the **open weights available today (Drex Max v1 and d1-2B)** on genuinely post-training, zero-contamination data: **560 arXiv Computer Science papers submitted Nov 1–8, 2026**, routed into **8 arXiv categories** from **title + abstract alone** (`96` stratified calibration-fit papers + `464` frozen test papers).

---

## Key Takeaways (`n = 464` Unseen Test Papers)

1. **The `>=93% -> 97%` rule holds out of the box for Drex Max v1 — even slightly better than advertised.**
   Drex Max v1 clears `>= 0.93` on **33.6%** (`156/464`) of test papers by raw top probability (`p_top`) and hits **98.7%** `[96.7%, 100.0%]` accuracy (`154/156`). Under Drex's stricter chance-adjusted `answer()` score (`chance_corrected >= 0.93`, equivalent to `p_top >= 0.93875`), it clears **31.0%** (`144/464`) at **98.6%** `[96.5%, 100.0%]` accuracy (`142/144`). The only **2 misses above 93%** (`2611.02152` on backpropagation derivation filed under `cs.CL`, and `2611.08108` on post-quantum Ethereum signatures filed under `cs.DC`) are author-filing quirks that d1-2B also missed.
2. **Drex Max v1 routes 68% of papers cleanly at 95% accuracy; d1-2B manages 29%.**
   Ranking predictions from highest to lowest confidence, Drex Max v1 auto-sorts **68.1%** (`316/464`, 95% CI `[50.9%, 78.2%]`) of test papers while maintaining `>= 95.0%` primary accuracy (`p_top >= 0.715`), compared to **29.3%** (`136/464`, `[4.7%, 51.9%]`) for d1-2B (`p_top >= 0.757`). Including valid author cross-listed tags raises Drex's auto-sort accuracy even higher (`88.1%` overall vs. `84.9%` primary).
3. **Out-of-the-box d1-2B is heavily underconfident — not overconfident — and a 96-label scalar fix nearly eliminates its calibration gap.**
   Out of the box, d1-2B leaves most correct predictions in the `60–85%` confidence band (15-bin ECE **11.7%** vs. Drex's **5.6%**), so only **4.5%** (`21/464`) reach `p_top >= 0.93` (where `21/21 = 100.0%` are right, exact Clopper-Pearson `[83.9%, 100.0%]`). Fitting a single scalar temperature multiplier on **96 labeled examples (12/category)** (`τ = 0.701`) slashes d1-2B's ECE nearly **3x** (`11.7% -> 4.3%`) and **4.1x expands** `p_top >= 0.93` coverage (`4.5% -> 18.8%`, `87/464`) at **96.6%** accuracy `[92.1%, 100.0%]`. On Drex Max v1 (`τ = 0.916`), ECE drops from **5.6% -> 4.0%** and `p_top >= 0.93` coverage increases from **33.6% -> 37.7%** (`175/464`) at **98.9%** accuracy (`173/175`).
4. **Confidence breaks gracefully at real category overlaps and never flips above 93% on Drex.**
   Reversing the order of all 8 option choices flips **0 / 156 (0.0%)** Drex predictions at `p_top >= 0.93` and **3.9%** (`18/464`) overall (all 18 had `p_top < 0.60`, mean `45.9%`). Accuracy is highest on sharp domains (`cs.CV`: `98.3%`, `cs.RO`: `91.4%`, `cs.CL`: `91.4%`) and weakest on **`cs.SE` (`70.7%` Drex / `63.8%` d1)** and **`cs.DC` (`74.1%` Drex / `50.0%` d1)** where LLM-agent and distributed-ML papers blend into `cs.CL` and `cs.LG`. On cross-listed papers (`n = 63`), Drex's primary match drops from `86.5%` to `74.6%`, yet its **any-listed accuracy is `98.4%`** (`62/63`) and its `>=93%` rate drops by half (`36.2% -> 17.5%`).

---

## Important Note: The Two "Confidence" Definitions

Drex Max and d1-2B compute `confidence` differently in their reference Python wrappers:

- **Raw top probability (`p_top`)** — returned directly by **`LiquidAI/d1-2B`** (`SystemOneModel.system_one_batch()` in the HuggingFace repo):
  $$\text{confidence}_{\text{raw}} = p_{\text{top}} = \max_{k \in \{1,\dots,8\}} p_k$$
- **Chance-corrected confidence (`chance_corrected`)** — returned by **`answer()` in [`nace-ai/drex-max-v1/code/drex_max/model.py`](https://huggingface.co/nace-ai/drex-max-v1/blob/main/code/drex_max/model.py)**:
  $$\text{confidence}_{\text{cc}} = \frac{p_{\text{top}} - \frac{1}{K}}{1 - \frac{1}{K}} = \frac{p_{\text{top}} - 0.125}{0.875} \quad (K = 8)$$

Clearing `chance_corrected >= 0.93` requires a stricter raw top probability of **`p_top >= 0.93875`** (`93.88%`). To ensure an apples-to-apples comparison, **all evaluations below report both definitions side-by-side** (using `p_top` as the primary probability metric for Expected Calibration Error against empirical accuracy). Both checkpoints are evaluated using their native non-generative decision APIs and served checkpoint temperatures (`1.0087` on Drex Max v1; `1.3374` on d1-2B).

---

## Headline Results (`n = 464` Test Split)

All metrics are computed strictly on the held-out **test split (`n = 464`, 58 per category)** with **2,000 percentile bootstrap resamples (`seed = 0`)** for 95% confidence intervals (and exact Clopper-Pearson binomial intervals when accuracy is `100%`).

### 1. Classification Accuracy & Throughput

| Model | Primary Accuracy | Macro-F1 | Any-Listed Accuracy (incl. cross-lists) | Inference Speed & Hardware |
|---|---|---|---|---|
| **Drex Max v1 (27B)** | **84.9%** `[81.5%, 87.9%]` (`394/464`) | **85.2%** `[81.8%, 88.0%]` | **88.1%** `[84.9%, 90.9%]` (`409/464`) | `5.38 papers/s` `[5.33, 5.42]` (`A100-SXM4-80GB`, `bf16`, `batch=8`) |
| **d1-2B (1.9B)** | **76.1%** `[72.2%, 79.7%]` (`353/464`) | **76.3%** `[72.4%, 79.8%]` | **79.3%** `[75.6%, 83.0%]` (`368/464`) | `0.82 papers/s` `[0.81, 0.83]` (`Apple M3 MPS`, `bf16`, `batch=16`) |

### 2. Testing the `>=93% Sure -> 97% Right` Claim

| Model | Regime | Confidence Rule | Share Clearing `>=0.93` | Accuracy at `>=0.93` (Primary) | Accuracy at `>=0.93` (Any Listed) |
|---|---|---|---|---|---|
| **Drex Max v1 (27B)** | Out-of-the-box | `p_top >= 0.93` | **33.6%** `[29.3%, 37.7%]` (`156/464`) | **98.7%** `[96.7%, 100.0%]` (`154/156`) | **98.7%** (`154/156`) |
| **Drex Max v1 (27B)** | Out-of-the-box | `chance_corrected >= 0.93` (`p_top >= 0.939`) | **31.0%** `[26.7%, 35.1%]` (`144/464`) | **98.6%** `[96.5%, 100.0%]` (`142/144`) | **98.6%** (`142/144`) |
| **Drex Max v1 (27B)** | After 96-label fix | `p_top >= 0.93` | **37.7%** `[33.4%, 42.0%]` (`175/464`) | **98.9%** `[97.0%, 100.0%]` (`173/175`) | **98.9%** (`173/175`) |
| **Drex Max v1 (27B)** | After 96-label fix | `chance_corrected >= 0.93` | **36.2%** `[31.9%, 40.5%]` (`168/464`) | **98.8%** `[96.9%, 100.0%]` (`166/168`) | **98.8%** (`166/168`) |
| **d1-2B (1.9B)** | Out-of-the-box | `p_top >= 0.93` | **4.5%** `[2.8%, 6.5%]` (`21/464`) | **100.0%** (`21/21`, CP `[83.9%, 100.0%]`) | **100.0%** (`21/21`) |
| **d1-2B (1.9B)** | Out-of-the-box | `chance_corrected >= 0.93` | **3.4%** `[1.9%, 5.2%]` (`16/464`) | **100.0%** (`16/16`, CP `[79.4%, 100.0%]`) | **100.0%** (`16/16`) |
| **d1-2B (1.9B)** | After 96-label fix | `p_top >= 0.93` | **18.8%** `[15.5%, 22.4%]` (`87/464`) | **96.6%** `[92.1%, 100.0%]` (`84/87`) | **96.6%** (`84/87`) |
| **d1-2B (1.9B)** | After 96-label fix | `chance_corrected >= 0.93` | **16.8%** `[13.6%, 20.3%]` (`78/464`) | **96.2%** `[91.4%, 100.0%]` (`75/78`) | **96.2%** (`75/78`) |

![At 93%+ stated confidence, how often is each model actually right?](assets/claim_93.png)

### 3. Effect of the 96-Label Temperature Fix

One scalar logit multiplier $\tau$ per model is fitted by minimizing NLL on the disjoint **fit split (`n = 96`, 12 papers per category)** (`probabilities ** (1 / tau)` renormalized, equivalent to scaling log-probabilities before softmax). Both models had $\tau < 1.0$, meaning both were **underconfident** on unseen arXiv abstracts out of the box (Drex mildly so; d1-2B strongly so):

| Model | Fitted $\tau$ (`n = 96`) | 15-Bin ECE (`p_top`) Before -> After | Test NLL Before -> After | `chance_corrected` Gap Before -> After |
|---|---|---|---|---|
| **Drex Max v1 (27B)** | **0.916** `[0.651, 1.155]` | **5.6%** `[4.7%, 9.3%]` -> **4.0%** `[3.4%, 7.8%]` | `0.451` `[0.384, 0.526]` -> `0.447` `[0.375, 0.527]` | `8.4%` `[6.5%, 11.6%]` -> `6.4%` `[5.2%, 9.9%]` |
| **d1-2B (1.9B)** | **0.701** `[0.530, 0.854]` | **11.7%** `[9.6%, 15.5%]` -> **4.3%** `[3.8%, 8.9%]` | `0.685` `[0.628, 0.748]` -> `0.629` `[0.552, 0.714]` | `16.1%` `[13.4%, 19.9%]` -> `5.5%` `[5.0%, 10.3%]` |

| Drex Max v1 Reliability | d1-2B Reliability |
|---|---|
| ![Drex Max Reliability](assets/reliability_drex.png) | ![d1-2B Reliability](assets/reliability_d1.png) |

---

## Where Confidence Holds and Where It Breaks

### Coverage vs. Precision (Auto-Routing Without Human Review)

![Auto-sorting coverage vs precision](assets/coverage_precision.png)

Even after temperature scaling aligns d1-2B's probabilities with actual accuracy (`ECE = 4.3%`), **ranking discrimination remains much stronger in the 27B model**:
- **Drex Max v1 (27B)** safely auto-sorts **68.1%** (`316/464`, `[50.9%, 78.2%]`) at `>= 95.0%` primary accuracy (`p_top >= 0.715` out of the box; `0.747` after temp fix).
- **d1-2B (1.9B)** safely auto-sorts **29.3%** (`136/464`, `[4.7%, 51.9%]`) at `>= 95.0%` primary accuracy (`p_top >= 0.757` out of the box; `0.859` after temp fix).

### Out-of-the-Box Confidence Bands (`n = 464`)

| `p_top` Band | Drex Max v1 Count | Drex Primary / Any-Listed Acc. | d1-2B Count | d1 Primary / Any-Listed Acc. |
|---|---|---|---|---|
| `[0.93, 1.00]` | `156 (33.6%)` | **98.7% / 98.7%** (`mean 97.3%`) | `21 (4.5%)` | **100.0% / 100.0%** (`mean 95.4%`) |
| `[0.85, 0.93)` | `76 (16.4%)` | **96.1% / 98.7%** (`mean 89.2%`) | `48 (10.3%)` | **93.8% / 93.8%** (`mean 88.8%`) |
| `[0.80, 0.85)` | `41 (8.8%)` | `87.8% / 87.8%` (`mean 82.5%`) | `29 (6.2%)` | `96.6% / 96.6%` (`mean 82.4%`) |
| `[0.70, 0.80)` | `51 (11.0%)` | `84.3% / 90.2%` (`mean 75.0%`) | `86 (18.5%)` | **93.0% / 95.3%** (`mean 75.3%`) |
| `[0.60, 0.70)` | `55 (11.9%)` | `78.2% / 85.5%` (`mean 64.2%`) | `98 (21.1%)` | **83.7% / 88.8%** (`mean 64.9%`) |
| `[0.00, 0.60)` | `85 (18.3%)` | `52.9% / 60.0%` (`mean 49.8%`) | `182 (39.2%)` | `53.3% / 57.7%` (`mean 47.5%`) |

### Option-Order Reversal & Cross-List Ambiguity (`n = 464`)

1. **Option-order reversal is confined to uncertain papers**:
   Reversing the order of all 8 option descriptions flipped **3.9%** `[2.2%, 5.6%]` (`18/464`) of Drex Max v1 predictions (reversed accuracy `86.0%` `[82.8%, 89.0%]`) and **7.1%** `[4.7%, 9.5%]` (`33/464`) of d1-2B predictions (reversed accuracy `75.4%`). Crucially:
   - **Drex Max v1 (`>=0.93` / `0.60–0.93` / `<0.60`) flip rates**: **`0/156 (0.0%)`** / **`0/223 (0.0%)`** / **`18/85 (21.2%)`** (`mean p_top` on flipped papers `45.9%` vs. `80.9%` on steady papers).
   - **d1-2B (`>=0.93` / `0.60–0.93` / `<0.60`) flip rates**: **`0/21 (0.0%)`** / **`1/261 (0.4%)`** / **`32/182 (17.6%)`** (`mean p_top` on flipped papers `42.9%` vs. `66.7%` steady).
2. **Low confidence tracks genuine multi-category ambiguity**:
   On the **63 test papers** whose authors listed a secondary category inside our 8-label set (`n = 63` vs. `n = 401` single-set):
   - **Drex Max v1** primary accuracy drops from `86.5%` (`[83.1%, 89.7%]`) to `74.6%` (`[63.9%, 84.8%]`), yet **any-listed category accuracy hits `98.4%`** (`[94.6%, 100.0%]`, `62/63`) while its mean confidence drops (`80.5% -> 74.0%`) and `p_top >= 0.93` rate halves (`36.2% -> 17.5%`).
   - **d1-2B** shows the exact same pattern: primary drops `76.8% -> 71.4%`, **any-listed accuracy hits `95.2%`** (`60/63`), and `p_top >= 0.93` drops from `5.2% -> 0.0%` (`0/63`).

### Where Category Boundaries Break (`cs.SE` and `cs.DC` vs. `cs.CL` / `cs.LG`)

| Drex Max v1 Confusion Matrix | d1-2B Confusion Matrix |
|---|---|
| ![Drex Max Confusion Matrix](assets/confusion_drex.png) | ![d1-2B Confusion Matrix](assets/confusion_d1.png) |

Per-category primary recall (`n = 58` test papers each) reveals clear attractors:
- **Sharpest domains**: `cs.CV` (**98.3%** Drex / `89.7%` d1), `cs.RO` (**91.4%** Drex / **94.8%** d1), `cs.CL` (**91.4%** Drex / `81.0%` d1), `cs.LG` (**86.2%** Drex / `77.6%` d1), `cs.CR` (**84.5%** Drex / `81.0%` d1), `cs.HC` (**82.8%** Drex / `70.7%` d1).
- **Weakest domains**: **`cs.SE`** (**70.7%** Drex / `63.8%` d1) and **`cs.DC`** (**74.1%** Drex / `50.0%` d1). Both models over-predict `cs.CL` (`84` Drex / `91` d1 vs. `58` true) and `cs.LG` (`67` Drex / `75` d1) when routing LLM-for-software-engineering, multi-agent systems, distributed training, or cryptographic blockchain papers (`19%` of `cs.SE` goes to `cs.CL` on Drex; `26%` of `cs.DC` goes to `cs.CL` and `16%` to `cs.LG` on d1-2B).

---

## Methodology, Scope & Limitations

- **Unseen dataset (`Nov 1–8, 2026`)**: Pulled directly via `export.arxiv.org/api/query` across 8 CS categories (`cs.CL, cs.CV, cs.LG, cs.RO, cs.CR, cs.SE, cs.DC, cs.HC`), filtered to papers with `<= 1` cross-list (`11.81%` in-set cross-list rate overall across `2,142` eligible submissions, chosen over Set A containing `cs.AI` which had `27.92%` overlap). Stratified `seed = 0` sample of `70/category` (`560` total -> `96` fit + `464` test). Both models were released earlier and could not have trained on these abstracts.
- **Noisy author labels**: Gold labels reflect submitting authors' primary filing choice rather than an objective committee label—explaining why cross-listed papers have `98.4%` any-listed accuracy on Drex and why the only two Drex errors above 93% (`2611.02152` on backpropagation and `2611.08108` on post-quantum Ethereum signatures) are misfiled/dual-domain submissions.
- **Single-task scope & Nace benchmark comparison**: This study evaluates **one zero-shot task** (8-way CS abstract classification, `n = 464`). Differences vs. Nace's 37-task Decision Index report reflect domain/task characteristics and arXiv filing ambiguity rather than a broad statement on all decision tasks.
- **Model versions tested**: Tests open **Drex Max v1** (`nace-ai/drex-max-v1`, commit `81cb1111d9ceb6fb54eb2e300890ec5380f175c0`) and **d1-2B** (`LiquidAI/d1-2B`, commit `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`), not Nace's announced upcoming v1.1.

---

## Reproducing Everything

### Repository Layout

- [`scripts/build_dataset.py`](scripts/build_dataset.py) — arXiv API harvest (`data/raw/*.xml`), overlap check (`data/counts.json`), stratified fit/test split (`data/papers.jsonl`).
- [`scripts/common.py`](scripts/common.py) — Canonical & reversed 8-category prompts and shared output validation.
- [`scripts/run_d1.py`](scripts/run_d1.py) — Local `LiquidAI/d1-2B` execution via `SystemOneModel` (`results/d1_*.json[l]`).
- [`scripts/run_drex.py`](scripts/run_drex.py) & [`notebooks/drex_max_colab.ipynb`](notebooks/drex_max_colab.ipynb) — Cloud GPU `nace-ai/drex-max-v1` execution (`results/drex_*.json[l]`).
- [`scripts/analyze.py`](scripts/analyze.py) — Stratified test-split metrics, 96-label temperature fit, bootstrap CIs (`results/metrics.json`, `results/summary.md`).
- [`scripts/make_charts.py`](scripts/make_charts.py) — Publication PNGs under [`assets/`](assets/).
- [`NOTES.md`](NOTES.md) — Chronological lab notebook with exact software versions, timings, smoke checks, and limitations.

### Commands

```bash
# 1. Dataset + analysis environment (CPU)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_dataset.py --reuse-raw

# 2. LiquidAI/d1-2B (local Apple Silicon MPS or CUDA)
python3 -m venv .venv-d1
.venv-d1/bin/pip install -r requirements-d1.txt
.venv-d1/bin/python scripts/run_d1.py --mode full

# 3. nace-ai/drex-max-v1 (Colab A100-80GB / G4 / H100 — see notebooks/drex_max_colab.ipynb)
pip install -r weights/drex-max-v1/requirements.txt
python3 scripts/run_drex.py --weights weights/drex-max-v1 --papers data/papers.jsonl --results-dir results --batch-size 8 --mode full

# 4. Compute metrics & regenerate all 7 charts
.venv/bin/python scripts/analyze.py
```

---

## References

- Nace AI — *Towards Foundation Decision Models* (Nov 6, 2026): https://www.nace.ai/research/towards-foundation-decision-models
- Nace AI — *Drex Max v1.1 open weights announcement*: https://x.com/NaceAI/status/2119792132558819447
- Artificial Analysis — *Decision Index v2 calibration scoring*: https://x.com/ArtificialAnlys/status/2119578471559213411
- Open-weight checkpoints tested: [`nace-ai/drex-max-v1`](https://huggingface.co/nace-ai/drex-max-v1) & [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B)
