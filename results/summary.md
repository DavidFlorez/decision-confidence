# Decision Model Calibration on Unseen Nov 1–8, 2026 arXiv Papers — Test-Split Summary

All headline numbers are computed strictly on the stratified **test split (`n = 464`, 58 per category)** with **2,000 percentile bootstrap resamples (`seed = 0`)** for 95% confidence intervals. Temperature scaling uses one scalar multiplier $\tau$ per model fitted by NLL on the disjoint **fit split (`n = 96`, 12 per category)**.

## 1. Overall Accuracy, Any-Listed Accuracy, and Throughput (`n = 464`)

| Model | Primary Accuracy | Macro-F1 | Any-Listed Category Accuracy | Throughput (papers/s) |
|---|---|---|---|---|
| **Drex Max v1 (27B)** | 84.9% [81.5%, 87.9%] (n=464) | 85.2% [81.8%, 88.0%] (n=464) | 88.1% [84.9%, 90.9%] (n=464) | 5.38 [5.33, 5.42] (n=464) |
| **d1-2B (1.9B)** | 76.1% [72.2%, 79.7%] (n=464) | 76.3% [72.4%, 79.8%] (n=464) | 79.3% [75.6%, 83.0%] (n=464) | 0.82 [0.81, 0.83] (n=464) |

## 2. Calibration (15-Bin ECE) & 96-Label Temperature Scaling

> **Note on confidence definitions**: `p_top` is the raw top-class softmax probability after each checkpoint's served temperature (`d1-2B` native confidence). `chance_corrected = (p_top - 1/8) / (7/8)` removes the 12.5% 8-way uniform chance floor (`Drex Max` native confidence); because it is not a probability of correctness, its 'ECE' against accuracy is reported strictly for comparison against the `>=93% -> 97%` rule.

| Model | Fitted $\tau$ (`n=96`) | ECE (`p_top`, before) | ECE (`p_top`, after) | Test NLL (before -> after) | `chance_corrected` Gap (before -> after) |
|---|---|---|---|---|---|
| **Drex Max v1 (27B)** | 0.916 [0.651, 1.155] (n=96) | 5.6% [4.7%, 9.3%] (n=464) | 4.0% [3.4%, 7.8%] (n=464) | 0.451 [0.384, 0.526] (n=464) -> 0.447 [0.375, 0.527] (n=464) | 8.4% [6.5%, 11.6%] (n=464) -> 6.4% [5.2%, 9.9%] (n=464) |
| **d1-2B (1.9B)** | 0.701 [0.530, 0.854] (n=96) | 11.7% [9.6%, 15.5%] (n=464) | 4.3% [3.8%, 8.9%] (n=464) | 0.685 [0.628, 0.748] (n=464) -> 0.629 [0.552, 0.714] (n=464) | 16.1% [13.4%, 19.9%] (n=464) -> 5.5% [5.0%, 10.3%] (n=464) |

## 3. Testing the `>=93% Sure -> 97% Right` Claim (`n = 464`)

| Model | Regime | Definition | Share Clearing `>=0.93` | Accuracy at `>=0.93` (Primary) | Accuracy at `>=0.93` (Any Listed) |
|---|---|---|---|---|---|
| **Drex Max v1 (27B)** | Out-of-the-box | `p_top >= 0.93` | 33.6% [29.3%, 37.7%] (n=464) (156/464) | 98.7% [96.7%, 100.0%] (n=156) (154/156) | 98.7% [96.7%, 100.0%] (n=156) |
| **Drex Max v1 (27B)** | Out-of-the-box | `chance_corrected >= 0.93` (`p_top >= 0.939`) | 31.0% [26.7%, 35.1%] (n=464) (144/464) | 98.6% [96.5%, 100.0%] (n=144) (142/144) | 98.6% [96.5%, 100.0%] (n=144) |
| **Drex Max v1 (27B)** | After 96-paper temp fix | `p_top >= 0.93` | 37.7% [33.4%, 42.0%] (n=464) (175/464) | 98.9% [97.0%, 100.0%] (n=175) (173/175) | 98.9% [97.0%, 100.0%] (n=175) |
| **Drex Max v1 (27B)** | After 96-paper temp fix | `chance_corrected >= 0.93` (`p_top >= 0.939`) | 36.2% [31.9%, 40.5%] (n=464) (168/464) | 98.8% [96.9%, 100.0%] (n=168) (166/168) | 98.8% [96.9%, 100.0%] (n=168) |
| **d1-2B (1.9B)** | Out-of-the-box | `p_top >= 0.93` | 4.5% [2.8%, 6.5%] (n=464) (21/464) | 100.0% [100.0%, 100.0%] (n=21) *(exact CP: [83.9%, 100.0%])* (21/21) | 100.0% [100.0%, 100.0%] (n=21) |
| **d1-2B (1.9B)** | Out-of-the-box | `chance_corrected >= 0.93` (`p_top >= 0.939`) | 3.4% [1.9%, 5.2%] (n=464) (16/464) | 100.0% [100.0%, 100.0%] (n=16) *(exact CP: [79.4%, 100.0%])* (16/16) | 100.0% [100.0%, 100.0%] (n=16) |
| **d1-2B (1.9B)** | After 96-paper temp fix | `p_top >= 0.93` | 18.8% [15.5%, 22.4%] (n=464) (87/464) | 96.6% [92.1%, 100.0%] (n=87) (84/87) | 96.6% [92.1%, 100.0%] (n=87) |
| **d1-2B (1.9B)** | After 96-paper temp fix | `chance_corrected >= 0.93` (`p_top >= 0.939`) | 16.8% [13.6%, 20.3%] (n=464) (78/464) | 96.2% [91.4%, 100.0%] (n=78) (75/78) | 96.2% [91.4%, 100.0%] (n=78) |

## 4. Auto-Routing Coverage at `>=95%` Precision (`n = 464`)

| Model | Max Coverage at `>=95%` Precision | Accepted Papers | Cutoff (`p_top` / `chance_corrected` out-of-the-box) | Cutoff (`p_top` / `chance_corrected` after temp) |
|---|---|---|---|---|
| **Drex Max v1 (27B)** | 68.1% [50.9%, 78.2%] (n=464) | 316/464 | `0.715` / `0.675` | `0.747` / `0.711` |
| **d1-2B (1.9B)** | 29.3% [4.7%, 51.9%] (n=464) | 136/464 | `0.757` / `0.723` | `0.859` / `0.838` |

## 5. Option-Order Reversal Check (`n = 464`)

| Model | Answer Flip Rate | Reversed Primary Accuracy | Mean `p_top` (Flipped vs. Steady) | Flip Rate (`p_top < 0.60` / `0.60–0.93` / `>=0.93`) |
|---|---|---|---|---|
| **Drex Max v1 (27B)** | 3.9% [2.2%, 5.6%] (n=464) (18/464) | 86.0% [82.8%, 89.0%] (n=464) | 45.9% [42.9%, 48.9%] (n=18) vs. 80.9% [79.3%, 82.5%] (n=446) | 18/85 (21.2%) / 0/223 (0.0%) / 0/156 (0.0%) |
| **d1-2B (1.9B)** | 7.1% [4.7%, 9.5%] (n=464) (33/464) | 75.4% [71.3%, 79.1%] (n=464) | 42.9% [39.2%, 46.4%] (n=33) vs. 66.7% [65.1%, 68.2%] (n=431) | 32/182 (17.6%) / 1/261 (0.4%) / 0/21 (0.0%) |

## 6. Low Confidence vs. Label Ambiguity (In-Set Cross-Listed Papers)

| Model | Subset | Primary Accuracy | Any-Listed Accuracy | Mean `p_top` | Share `p_top >= 0.93` |
|---|---|---|---|---|---|
| **Drex Max v1 (27B)** | No in-set cross-list (`n=401`) | 86.5% [83.1%, 89.7%] (n=401) | 86.5% [83.1%, 89.7%] (n=401) | 80.5% [78.6%, 82.2%] (n=401) | 36.2% [31.6%, 40.7%] (n=401) |
| **Drex Max v1 (27B)** | 1 cross-list in Set B (`n=63`) | 74.6% [63.9%, 84.8%] (n=63) | 98.4% [94.6%, 100.0%] (n=63) | 74.0% [69.9%, 77.9%] (n=63) | 17.5% [8.6%, 28.1%] (n=63) |
| **d1-2B (1.9B)** | No in-set cross-list (`n=401`) | 76.8% [72.9%, 80.9%] (n=401) | 76.8% [72.9%, 80.9%] (n=401) | 65.4% [63.7%, 67.1%] (n=401) | 5.2% [3.2%, 7.6%] (n=401) |
| **d1-2B (1.9B)** | 1 cross-list in Set B (`n=63`) | 71.4% [59.7%, 82.3%] (n=63) | 95.2% [89.1%, 100.0%] (n=63) | 62.5% [59.0%, 66.0%] (n=63) | 0.0% [0.0%, 0.0%] (n=63) |

