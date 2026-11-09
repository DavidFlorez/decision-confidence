#!/usr/bin/env python3
"""Step 5 — Compute all calibration, accuracy, reorder, temperature-fix, and ambiguity metrics with 95% CIs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from common import CATEGORIES, NUM_OPTIONS  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 0
N_ECE_BINS = 15
CLAIM_THRESHOLD = 0.93
CLAIM_TARGET_ACC = 0.97
COVERAGE_TARGET_PREC = 0.95
PROB_FLOOR = 1e-15


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> list[float | None]:
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta_dist.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def make_ci_dict(point: float | None, boot_samples: np.ndarray, n: int, extra: dict | None = None) -> dict:
    valid = boot_samples[np.isfinite(boot_samples)]
    if point is None or (isinstance(point, float) and math.isnan(point)) or len(valid) == 0:
        res: dict = {"value": None, "ci95": [None, None], "n": int(n)}
    else:
        lo, hi = np.percentile(valid, [2.5, 97.5])
        res = {"value": float(point), "ci95": [float(lo), float(hi)], "n": int(n)}
    if extra:
        res.update(extra)
    return res


def compute_macro_f1_single(y_true: np.ndarray, y_pred: np.ndarray, k: int = NUM_OPTIONS) -> float:
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return float(np.mean(f1))


def compute_macro_f1_boot(y_true: np.ndarray, y_pred: np.ndarray, boot_idx: np.ndarray, k: int = NUM_OPTIONS) -> np.ndarray:
    b = boot_idx.shape[0]
    flat = y_true[boot_idx] * k + y_pred[boot_idx]  # (B, N)
    offsets = (np.arange(b)[:, None] * (k * k))
    counts = np.bincount((flat + offsets).ravel(), minlength=b * k * k).reshape(b, k, k)
    tp = np.diagonal(counts, axis1=1, axis2=2).astype(float)  # (B, K)
    fp = counts.sum(axis=1) - tp
    fn = counts.sum(axis=2) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return f1.mean(axis=1)


def compute_ece_and_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_ECE_BINS) -> tuple[float, list[dict]]:
    bin_ids = np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)
    n = len(conf)
    ece = 0.0
    bins_out: list[dict] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt > 0:
            mean_c = float(conf[mask].mean())
            acc_b = float(correct[mask].mean())
            gap = abs(acc_b - mean_c)
            ece += (cnt / n) * gap
        else:
            mean_c = None
            acc_b = None
            gap = None
        bins_out.append(
            {
                "bin": b,
                "low": float(lo),
                "high": float(hi),
                "count": cnt,
                "mean_conf": mean_c,
                "accuracy": acc_b,
                "abs_gap": gap,
            }
        )
    return float(ece), bins_out


def compute_ece_boot(conf: np.ndarray, correct: np.ndarray, boot_idx: np.ndarray, n_bins: int = N_ECE_BINS) -> np.ndarray:
    b_runs, n = boot_idx.shape
    bin_ids = np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)
    diffs = correct.astype(float) - conf  # (N,)

    b_bins = bin_ids[boot_idx]  # (B, N)
    b_diffs = diffs[boot_idx]   # (B, N)
    offsets = np.arange(b_runs)[:, None] * n_bins
    bin_diff_sums = np.bincount(
        (b_bins + offsets).ravel(),
        weights=b_diffs.ravel(),
        minlength=b_runs * n_bins,
    ).reshape(b_runs, n_bins)
    return np.abs(bin_diff_sums).sum(axis=1) / n


def coverage_at_target_precision(conf: np.ndarray, correct: np.ndarray, target: float = COVERAGE_TARGET_PREC) -> dict:
    """Largest fraction auto-accepted (highest confidence first, respecting ties) with accuracy >= target."""
    n = len(conf)
    # Sort descending by confidence; group exact ties together so threshold rule c >= t is exact
    order = np.argsort(-conf, kind="mergesort")
    s_conf = conf[order]
    s_corr = correct[order].astype(float)

    cum_corr = np.cumsum(s_corr)
    cum_n = np.arange(1, n + 1, dtype=float)
    # Only evaluate at the end of any tie run: where next confidence is strictly smaller (or last item)
    tie_ends = np.empty(n, dtype=bool)
    tie_ends[:-1] = s_conf[:-1] > s_conf[1:]
    tie_ends[-1] = True

    valid_ends = np.where(tie_ends & ((cum_corr / cum_n) >= target))[0]
    if len(valid_ends) == 0:
        return {
            "coverage": 0.0,
            "accepted_count": 0,
            "precision_at_cutoff": None,
            "threshold": None,
        }
    best_idx = int(valid_ends[-1])
    return {
        "coverage": float((best_idx + 1) / n),
        "accepted_count": int(best_idx + 1),
        "precision_at_cutoff": float(cum_corr[best_idx] / (best_idx + 1)),
        "threshold": float(s_conf[best_idx]),
    }


def coverage_at_precision_boot(conf: np.ndarray, correct: np.ndarray, boot_idx: np.ndarray, target: float = COVERAGE_TARGET_PREC) -> np.ndarray:
    out = np.zeros(boot_idx.shape[0], dtype=float)
    for i in range(boot_idx.shape[0]):
        idx = boot_idx[i]
        out[i] = coverage_at_target_precision(conf[idx], correct[idx], target=target)["coverage"]
    return out


def coverage_precision_curve(conf: np.ndarray, correct: np.ndarray) -> dict:
    order = np.argsort(-conf, kind="mergesort")
    s_conf = conf[order]
    s_corr = correct[order].astype(float)
    n = len(conf)
    cum_n = np.arange(1, n + 1)
    cum_prec = np.cumsum(s_corr) / cum_n
    return {
        "k": cum_n.tolist(),
        "coverage": (cum_n / n).tolist(),
        "precision": cum_prec.tolist(),
        "threshold": s_conf.tolist(),
    }


def fit_temperature(probs_fit: np.ndarray, y_fit: np.ndarray) -> tuple[float, float, float]:
    """Fit scalar temperature tau > 0 minimizing cross-entropy NLL on log(probs_fit)."""
    log_p = np.log(np.clip(probs_fit, PROB_FLOOR, 1.0))
    n = len(y_fit)
    rows = np.arange(n)

    def nll_at_log_tau(log_tau: float) -> float:
        tau = math.exp(log_tau)
        scaled_lp = log_softmax(log_p / tau, axis=1)
        return float(-np.mean(scaled_lp[rows, y_fit]))

    nll_before = nll_at_log_tau(0.0)
    res = minimize_scalar(nll_at_log_tau, bounds=(-4.0, 4.0), method="bounded", options={"xatol": 1e-10})
    tau_hat = float(math.exp(res.x))
    nll_after = float(res.fun)
    return tau_hat, nll_before, nll_after


def apply_temperature(probs: np.ndarray, tau: float) -> np.ndarray:
    log_p = np.log(np.clip(probs, PROB_FLOOR, 1.0))
    return softmax(log_p / tau, axis=1)


def evaluate_claim(
    conf: np.ndarray,
    correct: np.ndarray,
    correct_any: np.ndarray,
    boot_idx: np.ndarray,
    threshold: float = CLAIM_THRESHOLD,
) -> dict:
    n = len(conf)
    mask = conf >= threshold
    m = int(mask.sum())
    k_corr = int((correct & mask).sum())
    k_any = int((correct_any & mask).sum())

    share_pt = m / n
    share_boot = mask[boot_idx].mean(axis=1)

    acc_pt = (k_corr / m) if m > 0 else None
    any_pt = (k_any / m) if m > 0 else None

    b_mask = mask[boot_idx]
    b_m = b_mask.sum(axis=1)
    b_k = (correct[boot_idx] & b_mask).sum(axis=1)
    b_k_any = (correct_any[boot_idx] & b_mask).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        acc_boot = np.where(b_m > 0, b_k / b_m, np.nan)
        any_boot = np.where(b_m > 0, b_k_any / b_m, np.nan)

    return {
        "threshold": threshold,
        "claim_target": CLAIM_TARGET_ACC,
        "count_clearing": m,
        "correct_clearing": k_corr,
        "any_listed_correct_clearing": k_any,
        "share_clearing": make_ci_dict(share_pt, share_boot, n),
        "accuracy_at_threshold": make_ci_dict(
            acc_pt,
            acc_boot,
            m,
            extra={"clopper_pearson_ci95": clopper_pearson_ci(k_corr, m)},
        ),
        "any_listed_accuracy_at_threshold": make_ci_dict(
            any_pt,
            any_boot,
            m,
            extra={"clopper_pearson_ci95": clopper_pearson_ci(k_any, m)},
        ),
        "meets_97_point_estimate": bool(acc_pt is not None and acc_pt >= CLAIM_TARGET_ACC),
    }


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = row
    return out


def analyze_model(
    model_key: str,
    display_name: str,
    native_conf_type: str,
    served_temp_default: float,
    papers: list[dict],
    main_path: Path,
    rev_path: Path,
    meta_path: Path | None,
    boot_idx_test: np.ndarray,
    boot_idx_fit: np.ndarray,
) -> dict:
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    cat_set = set(CATEGORIES)

    main_rows = load_jsonl_by_id(main_path)
    rev_rows = load_jsonl_by_id(rev_path) if rev_path.exists() else {}
    run_meta = json.loads(meta_path.read_text(encoding="utf-8")) if (meta_path and meta_path.exists()) else {}

    fit_papers = [p for p in papers if p["split"] == "fit"]
    test_papers = [p for p in papers if p["split"] == "test"]
    assert len(fit_papers) == 96 and len(test_papers) == 464

    def extract_arrays(subset: list[dict], source_rows: dict[str, dict]) -> dict:
        n = len(subset)
        probs = np.zeros((n, NUM_OPTIONS), dtype=float)
        y_true = np.zeros(n, dtype=int)
        returned_conf = np.zeros(n, dtype=float)
        latencies = np.zeros(n, dtype=float)
        choices: list[str] = []

        for i, p in enumerate(subset):
            r = source_rows[p["id"]]
            vec = np.array([float(r["probabilities"][c]) for c in CATEGORIES], dtype=float)
            vec = vec / vec.sum()
            probs[i] = vec
            y_true[i] = cat_to_idx[p["primary"]]
            returned_conf[i] = float(r["confidence"])
            latencies[i] = float(r["latency"])
            choices.append(str(r["choice"]))

        y_pred = np.argmax(probs, axis=1)
        p_top = probs[np.arange(n), y_pred]
        cc = np.clip((p_top - 1.0 / NUM_OPTIONS) / (1.0 - 1.0 / NUM_OPTIONS), 0.0, 1.0)
        return {
            "probs": probs,
            "y_true": y_true,
            "y_pred": y_pred,
            "p_top": p_top,
            "chance_corrected": cc,
            "returned_conf": returned_conf,
            "latencies": latencies,
            "choices": choices,
        }

    fit_arr = extract_arrays(fit_papers, main_rows)
    test_arr = extract_arrays(test_papers, main_rows)

    # Verify returned confidence matches documented formula
    expected_native = test_arr["p_top"] if native_conf_type == "p_top" else test_arr["chance_corrected"]
    max_native_diff = float(np.max(np.abs(test_arr["returned_conf"] - expected_native)))

    correct = test_arr["y_pred"] == test_arr["y_true"]
    pred_cats = [CATEGORIES[idx] for idx in test_arr["y_pred"]]
    correct_any = np.array(
        [
            (pred == p["primary"]) or (pred in cat_set and pred in p["cross_lists"])
            for pred, p in zip(pred_cats, test_papers)
        ],
        dtype=bool,
    )

    # 1. Classification metrics
    acc_pt = float(correct.mean())
    acc_boot = correct[boot_idx_test].mean(axis=1)

    any_acc_pt = float(correct_any.mean())
    any_acc_boot = correct_any[boot_idx_test].mean(axis=1)

    f1_pt = compute_macro_f1_single(test_arr["y_true"], test_arr["y_pred"])
    f1_boot = compute_macro_f1_boot(test_arr["y_true"], test_arr["y_pred"], boot_idx_test)

    # Confusion matrix (8x8)
    cm_counts = np.bincount(
        test_arr["y_true"] * NUM_OPTIONS + test_arr["y_pred"],
        minlength=NUM_OPTIONS * NUM_OPTIONS,
    ).reshape(NUM_OPTIONS, NUM_OPTIONS)
    cm_row_norm = cm_counts / cm_counts.sum(axis=1, keepdims=True)

    # Per-category metrics
    per_category: dict[str, dict] = {}
    for idx, cat in enumerate(CATEGORIES):
        cat_mask = test_arr["y_true"] == idx
        cat_acc_pt = float(correct[cat_mask].mean())
        # Bootstrap per-category recall/accuracy
        b_mask = test_arr["y_true"][boot_idx_test] == idx
        b_cnt = b_mask.sum(axis=1)
        b_corr = (correct[boot_idx_test] & b_mask).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cat_acc_boot = np.where(b_cnt > 0, b_corr / b_cnt, np.nan)
        per_category[cat] = {
            "accuracy_recall": make_ci_dict(cat_acc_pt, cat_acc_boot, int(cat_mask.sum())),
            "predicted_count": int((test_arr["y_pred"] == idx).sum()),
            "mean_p_top": float(test_arr["p_top"][cat_mask].mean()),
            "mean_chance_corrected": float(test_arr["chance_corrected"][cat_mask].mean()),
        }

    # 2. Temperature scaling on 96-paper fit split
    tau_hat, fit_nll_before, fit_nll_after = fit_temperature(fit_arr["probs"], fit_arr["y_true"])
    tau_boot = np.array(
        [fit_temperature(fit_arr["probs"][idx], fit_arr["y_true"][idx])[0] for idx in boot_idx_fit],
        dtype=float,
    )
    probs_test_cal = apply_temperature(test_arr["probs"], tau_hat)
    p_top_cal = probs_test_cal[np.arange(len(test_papers)), test_arr["y_pred"]]
    cc_cal = np.clip((p_top_cal - 1.0 / NUM_OPTIONS) / (1.0 - 1.0 / NUM_OPTIONS), 0.0, 1.0)

    # Test NLL before & after
    rows_test = np.arange(len(test_papers))
    nll_vec_before = -log_softmax(np.log(np.clip(test_arr["probs"], PROB_FLOOR, 1.0)), axis=1)[rows_test, test_arr["y_true"]]
    nll_vec_after = -log_softmax(np.log(np.clip(probs_test_cal, PROB_FLOOR, 1.0)), axis=1)[rows_test, test_arr["y_true"]]

    # 3. ECE & Reliability diagrams (before and after temperature scaling, both definitions)
    ece_ptop_before, bins_ptop_before = compute_ece_and_bins(test_arr["p_top"], correct)
    ece_ptop_before_boot = compute_ece_boot(test_arr["p_top"], correct, boot_idx_test)

    ece_cc_before, bins_cc_before = compute_ece_and_bins(test_arr["chance_corrected"], correct)
    ece_cc_before_boot = compute_ece_boot(test_arr["chance_corrected"], correct, boot_idx_test)

    ece_ptop_after, bins_ptop_after = compute_ece_and_bins(p_top_cal, correct)
    ece_ptop_after_boot = compute_ece_boot(p_top_cal, correct, boot_idx_test)

    ece_cc_after, bins_cc_after = compute_ece_and_bins(cc_cal, correct)
    ece_cc_after_boot = compute_ece_boot(cc_cal, correct, boot_idx_test)

    ece_ptop_delta_boot = ece_ptop_after_boot - ece_ptop_before_boot

    # 4. The >=93% -> 97% claim (before and after temperature scaling, both definitions)
    claim_before_ptop = evaluate_claim(test_arr["p_top"], correct, correct_any, boot_idx_test)
    claim_before_cc = evaluate_claim(test_arr["chance_corrected"], correct, correct_any, boot_idx_test)
    claim_after_ptop = evaluate_claim(p_top_cal, correct, correct_any, boot_idx_test)
    claim_after_cc = evaluate_claim(cc_cal, correct, correct_any, boot_idx_test)

    # 5. Coverage at 95% precision (plus full precision-coverage curve)
    cov95_ptop = coverage_at_target_precision(test_arr["p_top"], correct)
    cov95_cc = coverage_at_target_precision(test_arr["chance_corrected"], correct)
    cov95_ptop_after = coverage_at_target_precision(p_top_cal, correct)
    cov95_cc_after = coverage_at_target_precision(cc_cal, correct)
    cov95_boot = coverage_at_precision_boot(test_arr["p_top"], correct, boot_idx_test)
    cov_curve = coverage_precision_curve(test_arr["p_top"], correct)
    cov_curve_any = coverage_precision_curve(test_arr["p_top"], correct_any)

    # 6. Low confidence vs. ambiguity (cross-list analysis on 464 test papers)
    in_set_xl = np.array([any(c in cat_set for c in p["cross_lists"]) for p in test_papers], dtype=bool)
    any_xl = np.array([len(p["cross_lists"]) > 0 for p in test_papers], dtype=bool)
    out_set_xl = any_xl & (~in_set_xl)
    no_xl_anywhere = ~any_xl

    def subset_stats(mask: np.ndarray) -> dict:
        m = int(mask.sum())
        if m == 0:
            return {"n": 0}
        b_m = mask[boot_idx_test]
        b_cnt = b_m.sum(axis=1)

        def boot_mean_masked(arr: np.ndarray) -> dict:
            pt = float(arr[mask].mean())
            b_sum = np.where(b_m, arr[boot_idx_test], 0.0).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                bs = np.where(b_cnt > 0, b_sum / b_cnt, np.nan)
            return make_ci_dict(pt, bs, m)

        return {
            "n": m,
            "share_of_test": m / len(test_papers),
            "primary_accuracy": boot_mean_masked(correct.astype(float)),
            "any_listed_accuracy": boot_mean_masked(correct_any.astype(float)),
            "mean_p_top": boot_mean_masked(test_arr["p_top"]),
            "mean_chance_corrected": boot_mean_masked(test_arr["chance_corrected"]),
            "share_p_top_ge_93": boot_mean_masked((test_arr["p_top"] >= CLAIM_THRESHOLD).astype(float)),
            "share_cc_ge_93": boot_mean_masked((test_arr["chance_corrected"] >= CLAIM_THRESHOLD).astype(float)),
        }

    # Bootstrap difference: (in_set_xl) minus (not in_set_xl)
    no_in_set_xl = ~in_set_xl
    b_in = in_set_xl[boot_idx_test]
    b_no = no_in_set_xl[boot_idx_test]
    with np.errstate(divide="ignore", invalid="ignore"):
        acc_diff_boot = (
            np.where(b_in, correct[boot_idx_test], 0.0).sum(axis=1) / b_in.sum(axis=1)
            - np.where(b_no, correct[boot_idx_test], 0.0).sum(axis=1) / b_no.sum(axis=1)
        )
        ptop_diff_boot = (
            np.where(b_in, test_arr["p_top"][boot_idx_test], 0.0).sum(axis=1) / b_in.sum(axis=1)
            - np.where(b_no, test_arr["p_top"][boot_idx_test], 0.0).sum(axis=1) / b_no.sum(axis=1)
        )
        cc_diff_boot = (
            np.where(b_in, test_arr["chance_corrected"][boot_idx_test], 0.0).sum(axis=1) / b_in.sum(axis=1)
            - np.where(b_no, test_arr["chance_corrected"][boot_idx_test], 0.0).sum(axis=1) / b_no.sum(axis=1)
        )

    ambiguity = {
        "in_set_crosslist": subset_stats(in_set_xl),
        "no_in_set_crosslist": subset_stats(no_in_set_xl),
        "diff_in_set_minus_none": {
            "primary_accuracy_diff": make_ci_dict(
                float(correct[in_set_xl].mean() - correct[no_in_set_xl].mean()),
                acc_diff_boot,
                len(test_papers),
            ),
            "mean_p_top_diff": make_ci_dict(
                float(test_arr["p_top"][in_set_xl].mean() - test_arr["p_top"][no_in_set_xl].mean()),
                ptop_diff_boot,
                len(test_papers),
            ),
            "mean_chance_corrected_diff": make_ci_dict(
                float(
                    test_arr["chance_corrected"][in_set_xl].mean()
                    - test_arr["chance_corrected"][no_in_set_xl].mean()
                ),
                cc_diff_boot,
                len(test_papers),
            ),
        },
        "three_way_breakdown": {
            "zero_crosslists_anywhere": subset_stats(no_xl_anywhere),
            "one_crosslist_outside_set": subset_stats(out_set_xl),
            "one_crosslist_inside_set": subset_stats(in_set_xl),
        },
    }

    # 7. Option-order reversal check (Step 4, on 464 test papers)
    option_order: dict = {}
    if rev_rows and all(p["id"] in rev_rows for p in test_papers):
        rev_arr = extract_arrays(test_papers, rev_rows)
        rev_correct = rev_arr["y_pred"] == rev_arr["y_true"]
        flipped = test_arr["y_pred"] != rev_arr["y_pred"]

        flip_pt = float(flipped.mean())
        flip_boot = flipped[boot_idx_test].mean(axis=1)

        rev_acc_pt = float(rev_correct.mean())
        rev_acc_boot = rev_correct[boot_idx_test].mean(axis=1)
        acc_delta_boot = rev_acc_boot - acc_boot

        rev_f1_pt = compute_macro_f1_single(rev_arr["y_true"], rev_arr["y_pred"])
        rev_f1_boot = compute_macro_f1_boot(rev_arr["y_true"], rev_arr["y_pred"], boot_idx_test)

        # Confidence of flipped vs steady papers (under canonical forward pass)
        n_flips = int(flipped.sum())
        n_steady = int((~flipped).sum())

        def masked_conf_ci(arr: np.ndarray, msk: np.ndarray) -> dict:
            cnt = int(msk.sum())
            if cnt == 0:
                return {"value": None, "ci95": [None, None], "n": 0}
            pt = float(arr[msk].mean())
            bm = msk[boot_idx_test]
            bc = bm.sum(axis=1)
            bsum = np.where(bm, arr[boot_idx_test], 0.0).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                bs = np.where(bc > 0, bsum / bc, np.nan)
            return make_ci_dict(pt, bs, cnt)

        # Flip rate across confidence bands
        bands = [
            ("low_lt_0.60", test_arr["p_top"] < 0.60),
            ("mid_0.60_to_0.93", (test_arr["p_top"] >= 0.60) & (test_arr["p_top"] < 0.93)),
            ("high_ge_0.93", test_arr["p_top"] >= 0.93),
        ]
        band_stats: dict[str, dict] = {}
        for bname, bmask in bands:
            bcnt = int(bmask.sum())
            f_in_b = int((flipped & bmask).sum())
            if bcnt > 0:
                bm = bmask[boot_idx_test]
                bc = bm.sum(axis=1)
                bf = (flipped[boot_idx_test] & bm).sum(axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fbs = np.where(bc > 0, bf / bc, np.nan)
                band_stats[bname] = {
                    "n": bcnt,
                    "flips": f_in_b,
                    "flip_rate": make_ci_dict(f_in_b / bcnt, fbs, bcnt),
                }
            else:
                band_stats[bname] = {"n": 0, "flips": 0, "flip_rate": {"value": None, "ci95": [None, None], "n": 0}}

        # Also check high chance_corrected >= 0.93 flip rate
        cc_hi_mask = test_arr["chance_corrected"] >= 0.93
        cc_hi_n = int(cc_hi_mask.sum())
        cc_hi_flips = int((flipped & cc_hi_mask).sum())
        bm_cc = cc_hi_mask[boot_idx_test]
        bc_cc = bm_cc.sum(axis=1)
        bf_cc = (flipped[boot_idx_test] & bm_cc).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fbs_cc = np.where(bc_cc > 0, bf_cc / bc_cc, np.nan)

        option_order = {
            "flip_count": n_flips,
            "steady_count": n_steady,
            "flip_rate": make_ci_dict(flip_pt, flip_boot, len(test_papers)),
            "reversed_primary_accuracy": make_ci_dict(rev_acc_pt, rev_acc_boot, len(test_papers)),
            "reversed_macro_f1": make_ci_dict(rev_f1_pt, rev_f1_boot, len(test_papers)),
            "accuracy_change_reversed_minus_canonical": make_ci_dict(
                rev_acc_pt - acc_pt, acc_delta_boot, len(test_papers)
            ),
            "mean_p_top_flipped": masked_conf_ci(test_arr["p_top"], flipped),
            "mean_p_top_steady": masked_conf_ci(test_arr["p_top"], ~flipped),
            "mean_cc_flipped": masked_conf_ci(test_arr["chance_corrected"], flipped),
            "mean_cc_steady": masked_conf_ci(test_arr["chance_corrected"], ~flipped),
            "flip_rate_by_p_top_band": band_stats,
            "flip_rate_at_cc_ge_0.93": {
                "n": cc_hi_n,
                "flips": cc_hi_flips,
                "flip_rate": make_ci_dict((cc_hi_flips / cc_hi_n) if cc_hi_n > 0 else None, fbs_cc, cc_hi_n),
            },
        }

    # 8. Throughput & Latency
    lat_pt = float(test_arr["latencies"].mean())
    lat_boot = test_arr["latencies"][boot_idx_test].mean(axis=1)
    tps_pt = 1.0 / lat_pt if lat_pt > 0 else None
    tps_boot = 1.0 / lat_boot

    full_run_meta = run_meta.get("full_run", {})
    throughput = {
        "hardware_env": run_meta.get("env", {}),
        "mean_latency_sec_test": make_ci_dict(lat_pt, lat_boot, len(test_papers)),
        "papers_per_sec_test": make_ci_dict(tps_pt, tps_boot, len(test_papers)),
        "full_560_wall_sec": full_run_meta.get("wall_sec"),
        "full_560_papers_per_sec": full_run_meta.get("papers_per_sec"),
    }

    return {
        "model_key": model_key,
        "display_name": display_name,
        "native_confidence_definition": native_conf_type,
        "served_checkpoint_temperature": served_temp_default,
        "max_native_confidence_verification_diff": max_native_diff,
        "n_fit": len(fit_papers),
        "n_test": len(test_papers),
        "classification": {
            "primary_accuracy": make_ci_dict(acc_pt, acc_boot, len(test_papers), extra={"correct": int(correct.sum())}),
            "macro_f1": make_ci_dict(f1_pt, f1_boot, len(test_papers)),
            "any_listed_category_accuracy": make_ci_dict(
                any_acc_pt, any_acc_boot, len(test_papers), extra={"correct": int(correct_any.sum())}
            ),
            "per_category": per_category,
            "confusion_matrix_categories": CATEGORIES,
            "confusion_matrix_counts": cm_counts.tolist(),
            "confusion_matrix_row_normalized": cm_row_norm.tolist(),
        },
        "calibration": {
            "mean_p_top_before": make_ci_dict(
                float(test_arr["p_top"].mean()), test_arr["p_top"][boot_idx_test].mean(axis=1), len(test_papers)
            ),
            "mean_chance_corrected_before": make_ci_dict(
                float(test_arr["chance_corrected"].mean()),
                test_arr["chance_corrected"][boot_idx_test].mean(axis=1),
                len(test_papers),
            ),
            "mean_p_top_after": make_ci_dict(
                float(p_top_cal.mean()), p_top_cal[boot_idx_test].mean(axis=1), len(test_papers)
            ),
            "mean_chance_corrected_after": make_ci_dict(
                float(cc_cal.mean()), cc_cal[boot_idx_test].mean(axis=1), len(test_papers)
            ),
            "ece_15bin_p_top_before": make_ci_dict(ece_ptop_before, ece_ptop_before_boot, len(test_papers)),
            "ece_15bin_p_top_after": make_ci_dict(ece_ptop_after, ece_ptop_after_boot, len(test_papers)),
            "ece_15bin_p_top_change": make_ci_dict(
                ece_ptop_after - ece_ptop_before, ece_ptop_delta_boot, len(test_papers)
            ),
            "ece_15bin_chance_corrected_before": make_ci_dict(ece_cc_before, ece_cc_before_boot, len(test_papers)),
            "ece_15bin_chance_corrected_after": make_ci_dict(ece_cc_after, ece_cc_after_boot, len(test_papers)),
            "nll_test_before": make_ci_dict(
                float(nll_vec_before.mean()), nll_vec_before[boot_idx_test].mean(axis=1), len(test_papers)
            ),
            "nll_test_after": make_ci_dict(
                float(nll_vec_after.mean()), nll_vec_after[boot_idx_test].mean(axis=1), len(test_papers)
            ),
            "reliability_bins_p_top_before": bins_ptop_before,
            "reliability_bins_p_top_after": bins_ptop_after,
            "reliability_bins_cc_before": bins_cc_before,
            "reliability_bins_cc_after": bins_cc_after,
        },
        "temperature_scaling": {
            "fit_split_n": len(fit_papers),
            "fitted_multiplier_tau": make_ci_dict(tau_hat, tau_boot, len(fit_papers)),
            "effective_total_temperature": served_temp_default * tau_hat,
            "fit_nll_before": fit_nll_before,
            "fit_nll_after": fit_nll_after,
        },
        "claim_93": {
            "before_temp": {
                "p_top": claim_before_ptop,
                "chance_corrected": claim_before_cc,
            },
            "after_temp": {
                "p_top": claim_after_ptop,
                "chance_corrected": claim_after_cc,
            },
        },
        "coverage_at_95_precision": {
            "coverage": make_ci_dict(cov95_ptop["coverage"], cov95_boot, len(test_papers)),
            "accepted_count": cov95_ptop["accepted_count"],
            "precision_at_cutoff": cov95_ptop["precision_at_cutoff"],
            "threshold_p_top_before": cov95_ptop["threshold"],
            "threshold_cc_before": cov95_cc["threshold"],
            "threshold_p_top_after": cov95_ptop_after["threshold"],
            "threshold_cc_after": cov95_cc_after["threshold"],
            "curve_primary": cov_curve,
            "curve_any_listed": cov_curve_any,
        },
        "option_order_check": option_order,
        "crosslist_ambiguity": ambiguity,
        "throughput": throughput,
    }


def fmt_pct(ci_obj: dict, decimals: int = 1) -> str:
    val = ci_obj.get("value")
    lo, hi = ci_obj.get("ci95", [None, None])
    n = ci_obj.get("n")
    if val is None:
        return f"N/A (n={n})"
    return f"{val * 100:.{decimals}f}% [{lo * 100:.{decimals}f}%, {hi * 100:.{decimals}f}%] (n={n})"


def fmt_num(ci_obj: dict, decimals: int = 3) -> str:
    val = ci_obj.get("value")
    lo, hi = ci_obj.get("ci95", [None, None])
    n = ci_obj.get("n")
    if val is None:
        return f"N/A (n={n})"
    return f"{val:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}] (n={n})"


def build_summary_markdown(metrics: dict) -> str:
    models = metrics["models"]
    lines: list[str] = []
    lines.append("# Decision Model Calibration on Unseen Nov 1–8, 2026 arXiv Papers — Test-Split Summary")
    lines.append("")
    lines.append(
        f"All headline numbers are computed strictly on the stratified **test split (`n = {metrics['n_test']}`, "
        f"58 per category)** with **{metrics['bootstrap_resamples']:,} percentile bootstrap resamples (`seed = {metrics['bootstrap_seed']}`)** "
        f"for 95% confidence intervals. Temperature scaling uses one scalar multiplier $\\tau$ per model fitted by NLL "
        f"on the disjoint **fit split (`n = {metrics['n_fit']}`, 12 per category)**."
    )
    lines.append("")

    # Section 1: Classification & Throughput
    lines.append("## 1. Overall Accuracy, Any-Listed Accuracy, and Throughput (`n = 464`)")
    lines.append("")
    lines.append("| Model | Primary Accuracy | Macro-F1 | Any-Listed Category Accuracy | Throughput (papers/s) |")
    lines.append("|---|---|---|---|---|")
    for m in models.values():
        cls = m["classification"]
        tp = m["throughput"]
        lines.append(
            f"| **{m['display_name']}** | {fmt_pct(cls['primary_accuracy'])} | "
            f"{fmt_pct(cls['macro_f1'])} | {fmt_pct(cls['any_listed_category_accuracy'])} | "
            f"{fmt_num(tp['papers_per_sec_test'], 2)} |"
        )
    lines.append("")

    # Section 2: Calibration & Temperature Fix
    lines.append("## 2. Calibration (15-Bin ECE) & 96-Label Temperature Scaling")
    lines.append("")
    lines.append(
        "> **Note on confidence definitions**: `p_top` is the raw top-class softmax probability after each checkpoint's "
        "served temperature (`d1-2B` native confidence). `chance_corrected = (p_top - 1/8) / (7/8)` removes the 12.5% "
        "8-way uniform chance floor (`Drex Max` native confidence); because it is not a probability of correctness, "
        "its 'ECE' against accuracy is reported strictly for comparison against the `>=93% -> 97%` rule."
    )
    lines.append("")
    lines.append("| Model | Fitted $\\tau$ (`n=96`) | ECE (`p_top`, before) | ECE (`p_top`, after) | Test NLL (before -> after) | `chance_corrected` Gap (before -> after) |")
    lines.append("|---|---|---|---|---|---|")
    for m in models.values():
        cal = m["calibration"]
        ts = m["temperature_scaling"]
        lines.append(
            f"| **{m['display_name']}** | {fmt_num(ts['fitted_multiplier_tau'], 3)} | "
            f"{fmt_pct(cal['ece_15bin_p_top_before'])} | {fmt_pct(cal['ece_15bin_p_top_after'])} | "
            f"{fmt_num(cal['nll_test_before'], 3)} -> {fmt_num(cal['nll_test_after'], 3)} | "
            f"{fmt_pct(cal['ece_15bin_chance_corrected_before'])} -> {fmt_pct(cal['ece_15bin_chance_corrected_after'])} |"
        )
    lines.append("")

    # Section 3: The >=93% -> 97% Claim
    lines.append("## 3. Testing the `>=93% Sure -> 97% Right` Claim (`n = 464`)")
    lines.append("")
    lines.append("| Model | Regime | Definition | Share Clearing `>=0.93` | Accuracy at `>=0.93` (Primary) | Accuracy at `>=0.93` (Any Listed) |")
    lines.append("|---|---|---|---|---|---|")
    for m in models.values():
        for reg_key, reg_label in [("before_temp", "Out-of-the-box"), ("after_temp", "After 96-paper temp fix")]:
            for def_key, def_label in [("p_top", "`p_top >= 0.93`"), ("chance_corrected", "`chance_corrected >= 0.93` (`p_top >= 0.939`)")]:
                c = m["claim_93"][reg_key][def_key]
                k_c = c["correct_clearing"]
                m_c = c["count_clearing"]
                acc_str = fmt_pct(c["accuracy_at_threshold"])
                cp = c["accuracy_at_threshold"].get("clopper_pearson_ci95", [None, None])
                if m_c > 0 and k_c == m_c and cp[0] is not None:
                    acc_str += f" *(exact CP: [{cp[0]*100:.1f}%, 100.0%])*"
                lines.append(
                    f"| **{m['display_name']}** | {reg_label} | {def_label} | "
                    f"{fmt_pct(c['share_clearing'])} ({m_c}/464) | "
                    f"{acc_str} ({k_c}/{m_c}) | "
                    f"{fmt_pct(c['any_listed_accuracy_at_threshold'])} |"
                )
    lines.append("")

    # Section 4: Coverage at 95% Precision
    lines.append("## 4. Auto-Routing Coverage at `>=95%` Precision (`n = 464`)")
    lines.append("")
    lines.append("| Model | Max Coverage at `>=95%` Precision | Accepted Papers | Cutoff (`p_top` / `chance_corrected` out-of-the-box) | Cutoff (`p_top` / `chance_corrected` after temp) |")
    lines.append("|---|---|---|---|---|")
    for m in models.values():
        cov = m["coverage_at_95_precision"]
        t_p0 = f"{cov['threshold_p_top_before']:.3f}" if cov["threshold_p_top_before"] is not None else "N/A"
        t_c0 = f"{cov['threshold_cc_before']:.3f}" if cov["threshold_cc_before"] is not None else "N/A"
        t_p1 = f"{cov['threshold_p_top_after']:.3f}" if cov["threshold_p_top_after"] is not None else "N/A"
        t_c1 = f"{cov['threshold_cc_after']:.3f}" if cov["threshold_cc_after"] is not None else "N/A"
        lines.append(
            f"| **{m['display_name']}** | {fmt_pct(cov['coverage'])} | "
            f"{cov['accepted_count']}/464 | `{t_p0}` / `{t_c0}` | `{t_p1}` / `{t_c1}` |"
        )
    lines.append("")

    # Section 5: Option-Order Reversal Check
    lines.append("## 5. Option-Order Reversal Check (`n = 464`)")
    lines.append("")
    lines.append("| Model | Answer Flip Rate | Reversed Primary Accuracy | Mean `p_top` (Flipped vs. Steady) | Flip Rate (`p_top < 0.60` / `0.60–0.93` / `>=0.93`) |")
    lines.append("|---|---|---|---|---|")
    for m in models.values():
        oo = m["option_order_check"]
        if not oo:
            continue
        b_lo = oo["flip_rate_by_p_top_band"]["low_lt_0.60"]
        b_md = oo["flip_rate_by_p_top_band"]["mid_0.60_to_0.93"]
        b_hi = oo["flip_rate_by_p_top_band"]["high_ge_0.93"]
        band_str = (
            f"{b_lo['flips']}/{b_lo['n']} ({(b_lo['flip_rate']['value'] or 0)*100:.1f}%) / "
            f"{b_md['flips']}/{b_md['n']} ({(b_md['flip_rate']['value'] or 0)*100:.1f}%) / "
            f"{b_hi['flips']}/{b_hi['n']} ({(b_hi['flip_rate']['value'] or 0)*100:.1f}%)"
        )
        lines.append(
            f"| **{m['display_name']}** | {fmt_pct(oo['flip_rate'])} ({oo['flip_count']}/464) | "
            f"{fmt_pct(oo['reversed_primary_accuracy'])} | "
            f"{fmt_pct(oo['mean_p_top_flipped'])} vs. {fmt_pct(oo['mean_p_top_steady'])} | {band_str} |"
        )
    lines.append("")

    # Section 6: Low Confidence vs. Cross-List Ambiguity
    lines.append("## 6. Low Confidence vs. Label Ambiguity (In-Set Cross-Listed Papers)")
    lines.append("")
    lines.append("| Model | Subset | Primary Accuracy | Any-Listed Accuracy | Mean `p_top` | Share `p_top >= 0.93` |")
    lines.append("|---|---|---|---|---|---|")
    for m in models.values():
        amb = m["crosslist_ambiguity"]
        for skey, slabel in [("no_in_set_crosslist", "No in-set cross-list"), ("in_set_crosslist", "1 cross-list in Set B")]:
            s = amb[skey]
            lines.append(
                f"| **{m['display_name']}** | {slabel} (`n={s['n']}`) | "
                f"{fmt_pct(s['primary_accuracy'])} | {fmt_pct(s['any_listed_accuracy'])} | "
                f"{fmt_pct(s['mean_p_top'])} | {fmt_pct(s['share_p_top_ge_93'])} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Drex Max v1 and d1-2B results on unseen arXiv test split.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow running with only available model outputs")
    parser.add_argument("--skip-charts", action="store_true", help="Skip generating PNG charts")
    args = parser.parse_args()

    papers_path = ROOT / "data" / "papers.jsonl"
    counts_path = ROOT / "data" / "counts.json"
    papers = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts_data = json.loads(counts_path.read_text(encoding="utf-8"))

    n_fit = sum(1 for p in papers if p["split"] == "fit")
    n_test = sum(1 for p in papers if p["split"] == "test")

    rng_test = np.random.default_rng(BOOT_SEED)
    boot_idx_test = rng_test.integers(0, n_test, size=(N_BOOT, n_test))

    rng_fit = np.random.default_rng(BOOT_SEED)
    boot_idx_fit = rng_fit.integers(0, n_fit, size=(N_BOOT, n_fit))

    specs = [
        {
            "key": "drex_max",
            "name": "Drex Max v1 (27B)",
            "native_conf": "chance_corrected",
            "served_temp": 1.0087417621345625,
            "main": ROOT / "results" / "drex_max.jsonl",
            "rev": ROOT / "results" / "drex_max_reversed.jsonl",
            "meta": ROOT / "results" / "drex_run_meta.json",
        },
        {
            "key": "d1_2b",
            "name": "d1-2B (1.9B)",
            "native_conf": "p_top",
            "served_temp": 1.337398453966105,
            "main": ROOT / "results" / "d1_2b.jsonl",
            "rev": ROOT / "results" / "d1_2b_reversed.jsonl",
            "meta": ROOT / "results" / "d1_run_meta.json",
        },
    ]

    models_out: dict[str, dict] = {}
    for sp in specs:
        if not sp["main"].exists():
            if args.allow_partial:
                print(f"[skip] {sp['main'].relative_to(ROOT)} not found yet (--allow-partial).", flush=True)
                continue
            raise FileNotFoundError(f"Missing required model output: {sp['main']}")
        print(f"[analyze] Evaluating {sp['name']} from {sp['main'].relative_to(ROOT)}...", flush=True)
        models_out[sp["key"]] = analyze_model(
            model_key=sp["key"],
            display_name=sp["name"],
            native_conf_type=sp["native_conf"],
            served_temp_default=sp["served_temp"],
            papers=papers,
            main_path=sp["main"],
            rev_path=sp["rev"],
            meta_path=sp["meta"],
            boot_idx_test=boot_idx_test,
            boot_idx_fit=boot_idx_fit,
        )

    metrics_payload = {
        "dataset": {
            "date_window": counts_data["date_window"],
            "categories": CATEGORIES,
            "set_b_overlap": counts_data["set_b"],
            "set_a_overlap": counts_data["set_a"],
        },
        "n_total": len(papers),
        "n_fit": n_fit,
        "n_test": n_test,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "ece_num_bins": N_ECE_BINS,
        "models": models_out,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"
    summary_path = results_dir / "summary.md"

    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    summary_md = build_summary_markdown(metrics_payload)
    summary_path.write_text(summary_md, encoding="utf-8")

    print(f"\nWrote {metrics_path.relative_to(ROOT)} and {summary_path.relative_to(ROOT)}.", flush=True)
    print("\n" + summary_md, flush=True)

    if not args.skip_charts and len(models_out) == 2:
        from make_charts import generate_all_charts

        generate_all_charts(metrics_payload, ROOT / "assets")


if __name__ == "__main__":
    main()
