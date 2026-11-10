# Decision Confidence Study — Research Notes

## Goal
Test whether open-weight decision models Drex Max v1 (Nace AI, 27B) and d1-2B (Liquid AI, 1.9B) honor Nace's headline calibration claim ("When Drex Max is at least 93% sure, it is right 97% of the time") on genuinely unseen data—560 arXiv papers submitted Nov 1–8, 2026 routed across 8 `cs.*` categories—and measure where confidence holds, where it breaks, and how much a 96-paper temperature fix helps.

## Setup
- **Models evaluated**:
  - **Drex Max v1** ([`nace-ai/drex-max-v1`](https://huggingface.co/nace-ai/drex-max-v1), SHA `81cb1111d9ceb6fb54eb2e300890ec5380f175c0`, 27B Qwen3.8-27B hybrid backbone with non-causal full attention every 4th layer + 255-way readout head, checkpoint `decision_config.json` temperature `1.0087417621345625`).
  - **d1-2B** ([`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B), SHA `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`, 1.92B LFM3-VL-2B backbone, checkpoint `calibration.json` default temperature `1.337398453966105`).
- **Context & citations**:
  - Nace AI technical report (Nov 6, 2026): [*Towards Foundation Decision Models: Drex*](https://www.nace.ai/research/towards-foundation-decision-models) ([arXiv:2611.07854](https://arxiv.org/abs/2611.07854)).
  - Artificial Analysis Decision Index v2 calibration scoring announcement: <https://x.com/ArtificialAnlys/status/2119578471559213411>.
  - Nace announcement of Drex Max v1.1 ("open weights soon"): <https://x.com/NaceAI/status/2119792132558819447>. This repo evaluates **Drex Max v1**, whose weights are open today.
  - Not included: LLM-router Part 2 (intentionally omitted per study design).
- **Dataset**:
  - Source: arXiv Atom API (`https://export.arxiv.org/api/query`), `submittedDate:[202611010000 TO 202611082359]`, raw XML committed under `data/raw/*.xml`.
  - Target 8 categories (**Set B**): `cs.CL`, `cs.CV`, `cs.LG`, `cs.RO`, `cs.CR`, `cs.SE`, `cs.DC`, `cs.HC`.
  - Filter: primary category in Set B and at most 1 cross-list across any arXiv archive (including non-`cs` archives).
  - Sample & split (`seed=0`, stratified): 70 papers/category = **560 total**, split into **96 fit** (12/category, for post-hoc temperature scaling) and **464 test** (58/category, for all headline metrics).
- **Confidence definitions** ($K = 8$):
  1. `p_top`: raw top-class softmax probability returned after each checkpoint's built-in temperature (`probs[choice]`; native `confidence` returned by `d1-2B`).
  2. `chance_corrected`: $(p_{\text{top}} - 1/8) / (1 - 1/8) = (p_{\text{top}} - 0.125) / 0.875$ (native `confidence` returned by Drex Max `answer()`). Clearing `chance_corrected >= 0.93` requires `p_top >= 0.93875` ($\approx 0.939$).
- **Hardware & environments**:
  - **MacBook Air (Apple M3, 16 GB RAM, macOS 26.7.1)**:
    - Dataset & analysis venv (`.venv`, Python 3.13.7): `numpy==2.5.4`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `matplotlib==3.11.2`, `defusedxml==0.7.1`.
    - `d1-2B` inference venv (`.venv-d1`, Python 3.13.7, Apple M3 `mps` `bfloat16`): `torch==2.15.0`, `transformers==5.22.1`, `torchvision==0.29.2`, `pillow==13.0.0`, `huggingface-hub==2.4.0`.
  - **Google Colab (`NVIDIA A100-SXM4-80GB` High-RAM, `79.25 GiB` VRAM, `167.05 GiB` RAM, `bf16`)** *(fallback after `G4` repeatedly returned `503 Service Unavailable`)*:
    - Dedicated Drex Max v1 environment (`Python 3.13.16`, `torch==2.14.1+cu130`, `CUDA 13.0`, `transformers==5.17.0`, `flash-linear-attention==0.6.0`, `kernels==0.18.2`).

## Log
- **2026-11-09 (Step 1 — Dataset pull & overlap check)**:
  - Queried `export.arxiv.org/api/query` across all 10 categories in `Set A ∪ Set B` (`cs.CL, cs.CV, cs.LG, cs.RO, cs.CR, cs.SE, cs.DC, cs.HC, cs.AI, cs.IR`) for `submittedDate:[202611010000 TO 202611082359]` with `max_results=2000` and 3.1 s delay between requests. Every category had `totalResults < 2000` (largest: `cs.LG` with 1,214 total entries including cross-lists; `cs.AI` with 1,169), so zero pagination truncation occurred. Saved raw XML to `data/raw/*.xml` (3,413 unique deduplicated papers).
  - Verified category-set overlap among primary-category papers with $\le 1$ cross-list:
    - **Set B** (`CL, CV, LG, RO, CR, SE, DC, HC`): **253 / 2,142 (11.81%)** cross-list into another label in Set B. Eligible counts per category: `cs.CL` 264, `cs.CV` 605, `cs.LG` 608, `cs.RO` 277, `cs.CR` 120, `cs.SE` 118, `cs.DC` 76, `cs.HC` 74.
    - **Set A** (`CL, CV, LG, AI, RO, CR, SE, IR`): **697 / 2,496 (27.92%)** cross-list into another label in Set A (`cs.AI` contributes 438 primary papers and overlaps heavily with `cs.LG`, `cs.CL`, and `cs.CV`; `cs.IR` has 66).
  - Sampled 70 papers/category (`seed=0`, 560 total) and stratified-split into 96 `fit` (12/category) and 464 `test` (58/category) in `data/papers.jsonl` and `data/counts.json`.
- **2026-11-09 (Step 2 & Step 4 — `LiquidAI/d1-2B` on Apple M3 `mps` `bfloat16`)**:
  - Verified remote code at snapshot `652f920e544ea37f81e4b0ce89b81d4ed0da90b1`: initially our MRO check looked for `SystemOneApi` directly on `D1Model.__mro__`, which failed because `D1Model` inherits from `SystemOneModel` (`modeling_d1.py`) and delegates `system_one_batch` via a `@cached_property` `model.engine` (`SystemOne`, which inherits from `SystemOneApi` in `api.py` and lazily loads `calibration.json` with `default = 1.337398453966105`). Updated `scripts/run_d1.py` to verify both `SystemOneModel` on `model` and `SystemOneApi.system_one_batch` on `model.engine`.
  - Smoke tests passed cleanly on `mps` (`bfloat16`, zero NaNs, no float32/CPU fallback needed): model card ticket returned `choice="billing"` (`p_billing = 0.7111`, `refund = 0.9970`, `topics = ["billing", "delivery"]`); 20-paper smoke test ran at `0.858 s/paper` (`max_prob_sum_abs_err = 3.33e-16`, `confidence == max_prob` exact, 18/20 right).
  - Completed full 560-paper canonical run (`676.2 s`, `0.83 papers/s`, `batch_size=16`) to `results/d1_2b.jsonl` and full 560-paper reversed option-order run (`684.8 s`, `0.82 papers/s`) to `results/d1_2b_reversed.jsonl`.
- **2026-11-09 (Step 3 & Step 4 — `nace-ai/drex-max-v1` on Google Colab `A100-SXM4-80GB`)**:
  - Provisioned via `colab` CLI after `G4` returned `503 Service Unavailable`. Downloaded `48.61 GiB` snapshot (`commit 81cb1111d9ceb6fb54eb2e300890ec5380f175c0`, base `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, `noncausal_full_attention`, `served_default_temperature = 1.0087417621345625`) in `139.5 s`.
  - Verified exact checkpoint match against H100 `bf16` reference via `inference.py` (`87.1 s`): `team="billing"` at `p=0.9988` (`diff = 2.68e-5`), `confidence=0.9982` (`diff = 4.03e-5`), `refund=0.9974` (`diff = 4.25e-6`).
  - 50-paper smoke test passed at `0.632 s/paper` (`wall = 31.6 s`, `max_prob_sum_abs_err = 1.11e-16`, `45/50` correct). Completed full 560 canonical (`103.7 s`, `5.40 papers/s`, `batch_size=8`) and 560 reversed (`104.4 s`, `5.37 papers/s`) runs to `results/drex_max.jsonl` and `results/drex_max_reversed.jsonl`, downloaded locally, and terminated the Colab VM immediately (`0` active sessions).
- **2026-11-09 (Step 5 & Step 6 — Calibration Analysis & Charts)**:
  - Ran `.venv/bin/python scripts/analyze.py` (`results/metrics.json`, `results/summary.md`) and generated/refined all 7 PNGs in `assets/` via `.venv/bin/python scripts/make_charts.py`, inspecting every chart image to eliminate label collisions and colorbar clipping.

## Key results
*(Strictly evaluated on the frozen `n = 464` test split [58/category], with 2,000-resample percentile bootstrap 95% CIs [`seed = 0`]; scalar temperatures fitted on disjoint `n = 96` fit split [12/category].)*

- **Overall classification (`n = 464`)**:
  - **Drex Max v1 (27B)**: **84.9%** `[81.5%, 87.9%]` exact primary accuracy (**85.2%** `[81.8%, 88.0%]` macro-F1; **88.1%** `[84.9%, 90.9%]` including cross-listed category), at `5.38` papers/s (`[5.33, 5.42]`) on `A100-SXM4-80GB`.
  - **d1-2B (1.9B)**: **76.1%** `[72.2%, 79.7%]` exact primary accuracy (**76.3%** `[72.4%, 79.8%]` macro-F1; **79.3%** `[75.6%, 83.0%]` including cross-listed category), at `0.82` papers/s (`[0.81, 0.83]`) on Apple M3 `mps` `bf16`.
- **Does `>=93% sure -> 97% right` hold?**
  - **Drex Max v1 holds — even stricter than claimed, across both definitions**:
    - Out-of-the-box `p_top >= 0.93`: clears **33.6%** `[29.3%, 37.7%]` (`156/464`), **98.7%** right `[96.7%, 100.0%]` (`154/156`).
    - Out-of-the-box `chance_corrected >= 0.93` (`answer()["confidence"]`, equivalent to `p_top >= 0.939`): clears **31.0%** `[26.7%, 35.1%]` (`144/464`), **98.6%** right `[96.5%, 100.0%]` (`142/144`).
    - Only **2 high-confidence misses** (`p_top >= 0.93`) occurred on all 464 unseen papers, and both look like ambiguous/misfiled author metadata (`2611.02152`, *"Detailed Derivation of The Backpropagation Algorithm"*, author-listed `cs.CL` vs. Drex `cs.LG` at `0.9767`; and `2611.08108`, *"Post-Quantum Authentication of IoT Data on Ethereum..."*, author-listed `cs.DC` vs. Drex `cs.CR` at `0.9861`). Both errors were also shared by `d1-2B`.
  - **d1-2B is severely underconfident out of the box, limiting `>=93%` coverage**:
    - Out-of-the-box `p_top >= 0.93`: clears only **4.5%** `[2.8%, 6.5%]` (`21/464`), **100.0%** right (`21/21`, exact Clopper-Pearson `[83.9%, 100.0%]`). With `chance_corrected >= 0.93`: **3.4%** (`16/464`), `16/16` right.
- **How much does a 96-label scalar temperature fix help?**
  - **d1-2B** (`τ = 0.701` `[0.530, 0.854]`, `n=96`): slashes 15-bin `p_top` ECE nearly **3x** from **11.7%** `[9.6%, 15.5%]` to **4.3%** `[3.8%, 8.9%]`, cuts Test NLL from `0.685` to `0.629`, and **4.1x expands** `p_top >= 0.93` coverage from **4.5%** (`21/464`) to **18.8%** `[15.5%, 22.4%]` (`87/464`) at **96.6%** accuracy `[92.1%, 100.0%]` (`84/87`).
  - **Drex Max v1** (`τ = 0.916` `[0.651, 1.155]`, `n=96`): reduces 15-bin `p_top` ECE from **5.6%** `[4.7%, 9.3%]` (mild mid-range underconfidence: `89.2%` mean confidence on `[0.85, 0.93)` achieves `96.1%` primary / `98.7%` any-listed accuracy) to **4.0%** `[3.4%, 7.8%]`, lifting `p_top >= 0.93` coverage from `33.6%` (`156/464`) to **37.7%** (`175/464`) at **98.9%** accuracy (`173/175`).
- **Auto-routing coverage at `>=95%` precision (`n = 464`)**:
  - **Drex Max v1**: **68.1%** `[50.9%, 78.2%]` (`316/464`) at cutoff `p_top >= 0.715` (`chance_corrected >= 0.675` out-of-the-box; `0.747` after temp fix).
  - **d1-2B**: **29.3%** `[4.7%, 51.9%]` (`136/464`) at cutoff `p_top >= 0.757` (`chance_corrected >= 0.723` out-of-the-box; `0.859` after temp fix).
- **Where confidence breaks (`category boundaries, cross-lists, option reversal`)**:
  - **Option-order reversal**: **0 / 156 (0.0%)** of Drex `p_top >= 0.93` predictions flipped when reversing the 8 options; all `18/464` (**3.9%** `[2.2%, 5.6%]`) Drex flips and `32/33` of `d1-2B`'s `33/464` (**7.1%** `[4.7%, 9.5%]`) flips occurred on borderline papers (`p_top < 0.60`, mean flipped confidence `45.9%` vs. `80.9%` steady for Drex).
  - **Cross-list ambiguity**: On papers with a secondary Set B tag (`n = 63`), Drex primary accuracy drops from `86.5%` (`n=401`) to `74.6%`, while **any-listed accuracy hits `98.4%`** `[94.6%, 100.0%]` (`62/63`) and Drex naturally halves its `>=0.93` predictions (`36.2% -> 17.5%`).
  - **Category hot-spots**: Vision (`98.3%`), Robotics (`91.4%`), and Language (`91.4%`) separate cleanly on Drex, whereas **Software Eng. (`cs.SE`: `70.7%` Drex / `63.8%` d1)** and **Distributed Computing (`cs.DC`: `74.1%` Drex / `50.0%` d1)** lose papers into `cs.CL` (`19%` of `cs.SE` on Drex; `26%` of `cs.DC` on `d1-2B`) and `cs.LG` (`14%` of `cs.DC` on Drex; `16%` on `d1-2B`).

## Visuals
- [`assets/cover.png`](assets/cover.png) — 2000×800 overview pairing key test-split outcomes (`84.9%` vs. `76.1%`, `98.6%` at 93%+ confidence, `68%` vs. `29%` at 95% auto-sort precision) with precision-coverage curves.
- [`assets/reliability_drex.png`](assets/reliability_drex.png) — Drex Max v1 calibration curve and confidence bin counts before (`5.6` pt gap) and after the 96-paper temperature fix (`τ = 0.92`, `4.0` pt gap).
- [`assets/reliability_d1.png`](assets/reliability_d1.png) — d1-2B calibration curve showing heavy out-of-the-box underconfidence (`11.7` pt gap) nearly eliminated by `τ = 0.70` (`4.3` pt gap).
- [`assets/claim_93.png`](assets/claim_93.png) — Direct test of the `>=93% -> 97%` rule across both confidence formulas (`p_top` and `chance_corrected`) before and after temperature scaling, with 95% CIs.
- [`assets/coverage_precision.png`](assets/coverage_precision.png) — Auto-sorting coverage vs. accuracy curve (strict primary and cross-list-inclusive) highlighting the `95%` accuracy cutoff (`68.1%` vs. `29.3%`).
- [`assets/confusion_drex.png`](assets/confusion_drex.png) — 8×8 row-normalized agreement matrix for Drex Max v1 highlighting `cs.SE -> cs.CL` and `cs.DC -> cs.LG` boundary overlap.
- [`assets/confusion_d1.png`](assets/confusion_d1.png) — 8×8 row-normalized agreement matrix for d1-2B showing heavier spill from `cs.DC` and `cs.SE` into `cs.CL` / `cs.LG`.

## Surprises and limitations
- **Author-chosen primary labels are noisy**: The gold label is the submitting author's chosen primary arXiv category, so this benchmark tests *agreement with the author's primary filing choice* rather than an objective ground truth. Even after dropping `cs.AI` and restricting to $\le 1$ cross-list, 11.81% of eligible Set B papers explicitly cross-list into a second category inside our 8-label set—which is why we evaluate both strict primary accuracy and secondary "any listed category" accuracy, and stratify confidence by cross-list status.
- **Single-task scope**: This experiment tests one task—8-way zero-shot topic routing on English computer-science titles and abstracts ($n = 464$ test papers). Do not generalize these numbers to "decision models are/aren't calibrated" broadly.
- **Task mismatch vs. Nace's report**: Nace's "at least 93% sure $\to$ right 97% of the time" figure was measured across its 37-benchmark Decision Index suite. Any difference observed here reflects domain/task shift and arXiv label ambiguity on unseen Nov 2026 preprints, not a refutation of Nace's benchmark measurement.
- **Statistical precision**: Every test-split metric is reported with `n` and a 2,000-resample percentile bootstrap 95% confidence interval (`seed=0`), without rounding that alters qualitative conclusions.

## Open questions / next steps
- Does fitting multi-class Dirichlet or per-category temperature scaling (requiring larger calibration sets than 96 papers) reduce residual bin error on ambiguous boundary pairs (`cs.LG` vs. `cs.CV`/`cs.CL`)?
- Once Nace releases open weights for **Drex Max v1.1**, how much do calibration and option-order flip rate change on this exact frozen Nov 1–8, 2026 arXiv split?
