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
  - **Google Colab (NVIDIA RTX PRO 6000 Blackwell `G4`, 96 GB VRAM, `bf16`)**:
    - Dedicated Drex Max v1 environment (`transformers==5.17.0`, `flash-linear-attention>=0.5.2`, `torch>=2.14.0`): logged below once Step 3 completes.

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

## Key results
*(Populated as Steps 2–5 complete.)*

## Visuals
*(Populated in Step 6 with paths under `assets/` and one-line captions.)*

## Surprises and limitations
- **Author-chosen primary labels are noisy**: The gold label is the submitting author's chosen primary arXiv category, so this benchmark tests *agreement with the author's primary filing choice* rather than an objective ground truth. Even after dropping `cs.AI` and restricting to $\le 1$ cross-list, 11.81% of eligible Set B papers explicitly cross-list into a second category inside our 8-label set—which is why we evaluate both strict primary accuracy and secondary "any listed category" accuracy, and stratify confidence by cross-list status.
- **Single-task scope**: This experiment tests one task—8-way zero-shot topic routing on English computer-science titles and abstracts ($n = 464$ test papers). Do not generalize these numbers to "decision models are/aren't calibrated" broadly.
- **Task mismatch vs. Nace's report**: Nace's "at least 93% sure $\to$ right 97% of the time" figure was measured across its 37-benchmark Decision Index suite. Any difference observed here reflects domain/task shift and arXiv label ambiguity on unseen Nov 2026 preprints, not a refutation of Nace's benchmark measurement.
- **Statistical precision**: Every test-split metric is reported with `n` and a 2,000-resample percentile bootstrap 95% confidence interval (`seed=0`), without rounding that alters qualitative conclusions.

## Open questions / next steps
- Does fitting multi-class Dirichlet or per-category temperature scaling (requiring larger calibration sets than 96 papers) reduce residual bin error on ambiguous boundary pairs (`cs.LG` vs. `cs.CV`/`cs.CL`)?
- Once Nace releases open weights for **Drex Max v1.1**, how much do calibration and option-order flip rate change on this exact frozen Nov 1–8, 2026 arXiv split?
