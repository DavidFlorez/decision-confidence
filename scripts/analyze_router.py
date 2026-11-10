#!/usr/bin/env python3
"""Part 2 Statistical Analysis — LLM-Router benchmark (SPROUT-o3mini) & Part 1 d1->Drex cascade with 95% CIs."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from scipy.stats import beta as beta_dist
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze import apply_temperature, fit_temperature  # noqa: E402
from common import compute_confidences  # noqa: E402
from router_common import (  # noqa: E402
    PRIMARY_FRONTIER_MODEL,
    PRIMARY_SMALL_MODEL,
    ROUTER_OPTIONS,
    compute_binary_confidences,
)

N_BOOT = 2000
BOOT_SEED = 0
N_ECE_BINS = 15
CLAIM_THRESHOLD = 0.93
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


def compute_ece_and_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_ECE_BINS) -> tuple[float, list[dict]]:
    bin_ids = np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)
    n = len(conf)
    ece = 0.0
    bins_out: list[dict] = []
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt > 0:
            mc = float(conf[mask].mean())
            acc = float(correct[mask].mean())
            gap = abs(acc - mc)
            ece += (cnt / n) * gap
        else:
            mc = acc = gap = None
        bins_out.append(
            {
                "bin": b,
                "low": b / n_bins,
                "high": (b + 1) / n_bins,
                "count": cnt,
                "mean_conf": mc,
                "accuracy": acc,
                "abs_gap": gap,
            }
        )
    return float(ece), bins_out


def compute_ece_boot(conf: np.ndarray, correct: np.ndarray, boot_idx: np.ndarray, n_bins: int = N_ECE_BINS) -> np.ndarray:
    b_runs, n = boot_idx.shape
    bin_ids = np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)
    diffs = correct.astype(float) - conf
    b_bins = bin_ids[boot_idx]
    b_diffs = diffs[boot_idx]
    offsets = np.arange(b_runs)[:, None] * n_bins
    sums = np.bincount((b_bins + offsets).ravel(), weights=b_diffs.ravel(), minlength=b_runs * n_bins).reshape(b_runs, n_bins)
    return np.abs(sums).sum(axis=1) / n


def eval_threshold_claim(conf: np.ndarray, correct: np.ndarray, boot_idx: np.ndarray, thr: float = CLAIM_THRESHOLD) -> dict:
    mask = conf >= thr
    n_sub = int(mask.sum())
    n_tot = len(conf)
    cov = n_sub / n_tot
    if n_sub == 0:
        return {
            "threshold": thr,
            "accuracy": None,
            "accuracy_ci95_bootstrap": [None, None],
            "accuracy_ci95_clopper_pearson": [None, None],
            "coverage": 0.0,
            "count": 0,
            "correct_count": 0,
            "n_total": n_tot,
        }
    k_sub = int(correct[mask].sum())
    acc = k_sub / n_sub
    b_mask = mask[boot_idx]
    b_corr = (correct & mask)[boot_idx].sum(axis=1)
    b_cnt = b_mask.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b_acc = np.where(b_cnt > 0, b_corr / b_cnt, np.nan)
    valid = b_acc[np.isfinite(b_acc)]
    boot_ci = [float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))] if len(valid) else [None, None]
    return {
        "threshold": thr,
        "accuracy": float(acc),
        "accuracy_ci95_bootstrap": boot_ci,
        "accuracy_ci95_clopper_pearson": clopper_pearson_ci(k_sub, n_sub),
        "coverage": float(cov),
        "count": n_sub,
        "correct_count": k_sub,
        "n_total": n_tot,
    }


def coverage_at_target(conf: np.ndarray, correct: np.ndarray, target: float = COVERAGE_TARGET_PREC) -> dict:
    n = len(conf)
    order = np.argsort(-conf, kind="mergesort")
    s_conf = conf[order]
    s_corr = correct[order].astype(float)
    cum_corr = np.cumsum(s_corr)
    cum_n = np.arange(1, n + 1, dtype=float)
    tie_ends = np.empty(n, dtype=bool)
    tie_ends[:-1] = s_conf[:-1] > s_conf[1:]
    tie_ends[-1] = True
    valid = np.where(tie_ends & ((cum_corr / cum_n) >= target))[0]
    if len(valid) == 0:
        return {"coverage": 0.0, "accepted_count": 0, "precision": None, "threshold": None}
    idx = int(valid[-1])
    return {
        "coverage": float((idx + 1) / n),
        "accepted_count": int(idx + 1),
        "precision": float(cum_corr[idx] / (idx + 1)),
        "threshold": float(s_conf[idx]),
    }


def fit_best_threshold(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Find threshold theta in [0, 1] maximizing routing accuracy (score >= theta -> predict 1). Breaks ties closest to 0.5."""
    uniq = np.unique(scores)
    cands = [0.0]
    for a, b in zip(uniq[:-1], uniq[1:]):
        cands.append(0.5 * (float(a) + float(b)))
    cands.append(1.0 + 1e-6)
    best_acc = -1.0
    best_theta = 0.5
    for th in cands:
        pred = (scores >= th).astype(int)
        acc = float((pred == y).mean())
        if acc > best_acc + 1e-12 or (abs(acc - best_acc) <= 1e-12 and abs(th - 0.5) < abs(best_theta - 0.5)):
            best_acc = acc
            best_theta = float(th)
    return best_theta, best_acc


def compute_pareto_curve(
    router_score: np.ndarray,
    small_acc: np.ndarray,
    frontier_acc: np.ndarray,
    small_cost: np.ndarray,
    frontier_cost: np.ndarray,
) -> dict:
    """Sweep threshold on router_score (higher = send to frontier) and trace empirical Cost/1k vs Mean Quality.

    Ties are handled strictly monotonically by step thresholds at distinct score values.
    """
    n = len(router_score)
    # Candidate thresholds from >max(score) (0% frontier) down to min(score) (100% frontier)
    uniq = np.unique(router_score)
    thresholds = [float(uniq[-1]) + 1e-6] + [float(u) for u in reversed(uniq)]
    costs_1k: list[float] = []
    quals: list[float] = []
    frac_frontier: list[float] = []
    thetas: list[float] = []

    for th in thresholds:
        to_fr = router_score >= th
        q = np.where(to_fr, frontier_acc, small_acc)
        c = np.where(to_fr, frontier_cost, small_cost)
        costs_1k.append(float(1000.0 * c.mean()))
        quals.append(float(q.mean()))
        frac_frontier.append(float(to_fr.mean()))
        thetas.append(th)

    # Sort by increasing cost for trapezoidal integration
    order = np.argsort(costs_1k, kind="mergesort")
    c_sorted = np.array(costs_1k)[order]
    q_sorted = np.array(quals)[order]
    f_sorted = np.array(frac_frontier)[order]
    t_sorted = np.array(thetas)[order]

    c_min, c_max = float(c_sorted[0]), float(c_sorted[-1])
    q_min, q_max = float(q_sorted[0]), float(q_sorted[-1])
    cost_span = max(c_max - c_min, 1e-12)
    qual_gap = q_max - q_min

    # Average quality across cost span
    auc_cost_qual = float(np.trapezoid(q_sorted, c_sorted) / cost_span)
    # Normalized APGR: (mean router quality - Always Small) / (Always Frontier - Always Small)
    apgr = float((auc_cost_qual - q_min) / qual_gap) if abs(qual_gap) > 1e-12 else 0.0
    # Random router AUC = 0.5 * (q_min + q_max), APGR_random = 0.50
    gain_over_random_auc = float(apgr - 0.5)

    # Interpolated quality & PGR at budget milestones (25%, 50%, 75% of frontier cost)
    milestones = {}
    for pct in (25, 50, 75):
        target_c = (pct / 100.0) * c_max
        q_at_b = float(np.interp(target_c, c_sorted, q_sorted))
        q_rand = float(q_min + ((target_c - c_min) / cost_span) * qual_gap)
        pgr = float((q_at_b - q_min) / qual_gap) if abs(qual_gap) > 1e-12 else 0.0
        milestones[f"pct_{pct}"] = {
            "target_cost_per_1k": target_c,
            "quality": q_at_b,
            "random_quality": q_rand,
            "lift_over_random_pts": q_at_b - q_rand,
            "pgr": pgr,
        }

    return {
        "cost_per_1k": c_sorted.tolist(),
        "quality": q_sorted.tolist(),
        "frac_frontier": f_sorted.tolist(),
        "thresholds": t_sorted.tolist(),
        "apgr": apgr,
        "gain_over_random_apgr": gain_over_random_auc,
        "milestones": milestones,
    }


def eval_fixed_policy(
    to_frontier: np.ndarray,
    small_acc: np.ndarray,
    frontier_acc: np.ndarray,
    small_cost: np.ndarray,
    frontier_cost: np.ndarray,
    y_gold: np.ndarray,
    boot_idx: np.ndarray,
) -> dict:
    q = np.where(to_frontier, frontier_acc, small_acc)
    c_1k = 1000.0 * np.where(to_frontier, frontier_cost, small_cost)
    acc_route = (to_frontier.astype(int) == y_gold).astype(float)

    q_sm = float(small_acc.mean())
    q_fr = float(frontier_acc.mean())
    c_fr = float(1000.0 * frontier_cost.mean())

    q_val = float(q.mean())
    c_val = float(c_1k.mean())
    pgr_val = (q_val - q_sm) / (q_fr - q_sm) if abs(q_fr - q_sm) > 1e-12 else 0.0

    q_boot = q[boot_idx].mean(axis=1)
    c_boot = c_1k[boot_idx].mean(axis=1)
    r_boot = acc_route[boot_idx].mean(axis=1)
    pgr_boot = (q_boot - small_acc[boot_idx].mean(axis=1)) / np.maximum(
        frontier_acc[boot_idx].mean(axis=1) - small_acc[boot_idx].mean(axis=1), 1e-12
    )

    n = len(y_gold)
    return {
        "esc_rate": make_ci_dict(float(to_frontier.mean()), to_frontier.astype(float)[boot_idx].mean(axis=1), n),
        "routing_accuracy": make_ci_dict(float(acc_route.mean()), r_boot, n),
        "system_quality": make_ci_dict(q_val, q_boot, n),
        "cost_per_1k": make_ci_dict(c_val, c_boot, n),
        "cost_savings_vs_frontier": float(1.0 - c_val / c_fr) if c_fr > 0 else 0.0,
        "pgr": make_ci_dict(float(pgr_val), pgr_boot, n),
    }


def analyze_router_model(
    name: str,
    queries: list[dict],
    recs_main: dict[str, dict],
    recs_rev: dict[str, dict],
    boot_idx_test: np.ndarray,
) -> dict:
    fit_q = [q for q in queries if q["split"] == "fit"]
    test_q = [q for q in queries if q["split"] == "test"]
    n_fit, n_test = len(fit_q), len(test_q)

    def extract_arrays(q_list: list[dict], recs: dict[str, dict]):
        probs = np.zeros((len(q_list), 2), dtype=float)
        y = np.zeros(len(q_list), dtype=int)
        choices = np.zeros(len(q_list), dtype=int)
        p_top = np.zeros(len(q_list), dtype=float)
        cc = np.zeros(len(q_list), dtype=float)
        p_fr = np.zeros(len(q_list), dtype=float)
        s_sm = np.zeros(len(q_list), dtype=float)
        s_fr = np.zeros(len(q_list), dtype=float)
        c_sm = np.zeros(len(q_list), dtype=float)
        c_fr = np.zeros(len(q_list), dtype=float)
        fams = []
        for i, q in enumerate(q_list):
            rec = recs[q["id"]]
            p_vec = np.array([rec["probabilities"]["small"], rec["probabilities"]["frontier"]], dtype=float)
            p_vec = p_vec / p_vec.sum()
            probs[i] = p_vec
            y[i] = int(q["gold_needs_frontier"])
            ch, pt, cci, pf = compute_binary_confidences(rec["probabilities"])
            choices[i] = 0 if ch == "small" else 1
            p_top[i] = pt
            cc[i] = cci
            p_fr[i] = pf
            s_sm[i] = float(q["outcomes"][PRIMARY_SMALL_MODEL]["score"])
            s_fr[i] = float(q["outcomes"][PRIMARY_FRONTIER_MODEL]["score"])
            c_sm[i] = float(q["outcomes"][PRIMARY_SMALL_MODEL]["cost_usd"])
            c_fr[i] = float(q["outcomes"][PRIMARY_FRONTIER_MODEL]["cost_usd"])
            fams.append(q["family"])
        return probs, y, choices, p_top, cc, p_fr, s_sm, s_fr, c_sm, c_fr, np.array(fams)

    probs_f, y_f, ch_f, pt_f, cc_f, pfr_f, ssm_f, sfr_f, csm_f, cfr_f, fam_f = extract_arrays(fit_q, recs_main)
    probs_t, y_t, ch_t, pt_t, cc_t, pfr_t, ssm_t, sfr_t, csm_t, cfr_t, fam_t = extract_arrays(test_q, recs_main)
    probs_r, y_r, ch_r, pt_r, cc_r, pfr_r, *_ = extract_arrays(test_q, recs_rev)

    # 100-label fit: scalar temperature tau & optimal thresholds
    tau_hat, nll_fit_before, nll_fit_after = fit_temperature(probs_f, y_f)
    probs_f_cal = apply_temperature(probs_f, tau_hat)
    probs_t_cal = apply_temperature(probs_t, tau_hat)
    pfr_f_cal = probs_f_cal[:, 1]
    pfr_t_cal = probs_t_cal[:, 1]
    pt_t_cal = probs_t_cal.max(axis=1)
    cc_t_cal = np.clip((pt_t_cal - 0.5) / 0.5, 0.0, 1.0)

    theta_fit_raw, fit_acc_raw = fit_best_threshold(pfr_f, y_f)
    theta_fit_cal, fit_acc_cal = fit_best_threshold(pfr_f_cal, y_f)

    # Threshold on fit matching ~50% of frontier cost
    cfr_fit_mean = float(1000.0 * cfr_f.mean())
    uniq_f = np.unique(pfr_f)
    best_th_50 = 0.5
    best_diff = 1e9
    for th in uniq_f:
        to_f = pfr_f >= th
        cost_th = float(1000.0 * np.where(to_f, cfr_f, csm_f).mean())
        if abs(cost_th - 0.5 * cfr_fit_mean) < best_diff:
            best_diff = abs(cost_th - 0.5 * cfr_fit_mean)
            best_th_50 = float(th)

    # Discrimination metrics (AUROC / AUPRC / Log loss / Brier) with bootstrap CIs
    auroc_pt = float(roc_auc_score(y_t, pfr_t))
    auprc_pt = float(average_precision_score(y_t, pfr_t))
    brier_raw = float(np.mean((pfr_t - y_t) ** 2))
    brier_cal = float(np.mean((pfr_t_cal - y_t) ** 2))
    lp_t = np.log(np.clip(probs_t, PROB_FLOOR, 1.0))
    lp_t_cal = np.log(np.clip(probs_t_cal, PROB_FLOOR, 1.0))
    nll_raw = float(-np.mean(lp_t[np.arange(n_test), y_t]))
    nll_cal = float(-np.mean(lp_t_cal[np.arange(n_test), y_t]))

    b_auroc = np.empty(N_BOOT, dtype=float)
    b_auprc = np.empty(N_BOOT, dtype=float)
    b_apgr = np.empty(N_BOOT, dtype=float)
    b_pgr50 = np.empty(N_BOOT, dtype=float)
    b_nll_raw = np.empty(N_BOOT, dtype=float)
    b_nll_cal = np.empty(N_BOOT, dtype=float)
    b_brier_raw = np.empty(N_BOOT, dtype=float)
    b_brier_cal = np.empty(N_BOOT, dtype=float)

    pareto_main = compute_pareto_curve(pfr_t, ssm_t, sfr_t, csm_t, cfr_t)
    pareto_cal = compute_pareto_curve(pfr_t_cal, ssm_t, sfr_t, csm_t, cfr_t)

    for b in range(N_BOOT):
        idx = boot_idx_test[b]
        yb = y_t[idx]
        pb = pfr_t[idx]
        pbc = pfr_t_cal[idx]
        if 0 < yb.sum() < len(yb):
            b_auroc[b] = roc_auc_score(yb, pb)
            b_auprc[b] = average_precision_score(yb, pb)
        else:
            b_auroc[b] = np.nan
            b_auprc[b] = np.nan
        par_b = compute_pareto_curve(pb, ssm_t[idx], sfr_t[idx], csm_t[idx], cfr_t[idx])
        b_apgr[b] = par_b["apgr"]
        b_pgr50[b] = par_b["milestones"]["pct_50"]["pgr"]
        b_nll_raw[b] = float(-np.mean(lp_t[idx, yb]))
        b_nll_cal[b] = float(-np.mean(lp_t_cal[idx, yb]))
        b_brier_raw[b] = float(np.mean((pb - yb) ** 2))
        b_brier_cal[b] = float(np.mean((pbc - yb) ** 2))

    # Regimes evaluated strictly out-of-sample on n=500 test
    regimes = {
        "argmax_050": eval_fixed_policy(pfr_t >= 0.50, ssm_t, sfr_t, csm_t, cfr_t, y_t, boot_idx_test),
        "nace_080": eval_fixed_policy(pfr_t >= 0.80, ssm_t, sfr_t, csm_t, cfr_t, y_t, boot_idx_test),
        "fit100_raw_theta": eval_fixed_policy(pfr_t >= theta_fit_raw, ssm_t, sfr_t, csm_t, cfr_t, y_t, boot_idx_test),
        "fit100_temp_theta": eval_fixed_policy(pfr_t_cal >= theta_fit_cal, ssm_t, sfr_t, csm_t, cfr_t, y_t, boot_idx_test),
        "fit100_budget50_theta": eval_fixed_policy(pfr_t >= best_th_50, ssm_t, sfr_t, csm_t, cfr_t, y_t, boot_idx_test),
    }

    # Calibration checks for BOTH p_top and chance_corrected (before & after temp scaling at argmax decision)
    corr_argmax = (ch_t == y_t)
    ch_cal = (pfr_t_cal >= 0.5).astype(int)
    corr_cal = (ch_cal == y_t)
    # Also check selective claim evaluated on thresholded predicted option at theta_fit_raw? Argmax keeps symmetry with confidence definition
    calib = {}
    for variant, conf_arr, corr_arr in [
        ("raw_p_top", pt_t, corr_argmax),
        ("raw_chance_corrected", cc_t, corr_argmax),
        ("temp_p_top", pt_t_cal, corr_cal),
        ("temp_chance_corrected", cc_t_cal, corr_cal),
    ]:
        ece_val, bins_info = compute_ece_and_bins(conf_arr, corr_arr)
        ece_boot = compute_ece_boot(conf_arr, corr_arr, boot_idx_test)
        cov95 = coverage_at_target(conf_arr, corr_arr, target=COVERAGE_TARGET_PREC)
        cov95_boot = np.array([coverage_at_target(conf_arr[idx], corr_arr[idx], COVERAGE_TARGET_PREC)["coverage"] for idx in boot_idx_test])
        calib[variant] = {
            "ece": make_ci_dict(ece_val, ece_boot, n_test),
            "mean_conf": float(conf_arr.mean()),
            "claim_93": eval_threshold_claim(conf_arr, corr_arr, boot_idx_test, thr=CLAIM_THRESHOLD),
            "coverage_at_95pct_precision": make_ci_dict(cov95["coverage"], cov95_boot, n_test, cov95),
            "bins": bins_info,
        }

    # Option-order reversal sensitivity on test split (n=500)
    flips = (ch_t != ch_r).astype(float)
    flip_boot = flips[boot_idx_test].mean(axis=1)
    rev_acc = (ch_r == y_t).astype(float)
    rev_acc_boot = rev_acc[boot_idx_test].mean(axis=1)
    rev_pfr_diff = np.abs(pfr_t - pfr_r)

    # Per-family breakdown on test
    per_family = {}
    for fam in sorted(set(fam_t.tolist())):
        m = fam_t == fam
        per_family[fam] = {
            "n": int(m.sum()),
            "small_acc": float(ssm_t[m].mean()),
            "frontier_acc": float(sfr_t[m].mean()),
            "gold_frontier_frac": float(y_t[m].mean()),
            "pred_frontier_frac_050": float((pfr_t[m] >= 0.5).mean()),
            "pred_frontier_frac_tuned": float((pfr_t[m] >= theta_fit_raw).mean()),
            "mean_p_frontier": float(pfr_t[m].mean()),
            "routing_acc_050": float((ch_t[m] == y_t[m]).mean()),
            "routing_acc_tuned": float(((pfr_t[m] >= theta_fit_raw).astype(int) == y_t[m]).mean()),
            "auroc": float(roc_auc_score(y_t[m], pfr_t[m])) if 0 < y_t[m].sum() < m.sum() else None,
        }

    # Secondary LLM pairs evaluated on the exact same prompt complexity probabilities pfr_t!
    secondary_pairs = {}
    for sm_id, fr_id in [
        ("wxai-llama-3-1-8b-instruct", "openai-o3-mini"),
        ("openai-gpt-4o-mini", "openai-o3-mini"),
        ("wxai-llama-3-2-3b-instruct", "aws-claude-3-5-sonnet-v1"),
    ]:
        sec_ssm = np.array([float(q["outcomes"][sm_id]["score"]) for q in test_q])
        sec_sfr = np.array([float(q["outcomes"][fr_id]["score"]) for q in test_q])
        sec_csm = np.array([float(q["outcomes"][sm_id]["cost_usd"]) for q in test_q])
        sec_cfr = np.array([float(q["outcomes"][fr_id]["cost_usd"]) for q in test_q])
        sec_y = (sec_ssm < sec_sfr).astype(int)
        sec_par = compute_pareto_curve(pfr_t, sec_ssm, sec_sfr, sec_csm, sec_cfr)
        secondary_pairs[f"{sm_id}_to_{fr_id}"] = {
            "small_acc": float(sec_ssm.mean()),
            "frontier_acc": float(sec_sfr.mean()),
            "small_cost_per_1k": float(1000.0 * sec_csm.mean()),
            "frontier_cost_per_1k": float(1000.0 * sec_cfr.mean()),
            "auroc": float(roc_auc_score(sec_y, pfr_t)),
            "apgr": sec_par["apgr"],
            "milestones": sec_par["milestones"],
        }

    return {
        "model": name,
        "fit": {
            "n": n_fit,
            "tau": tau_hat,
            "nll_before": nll_fit_before,
            "nll_after": nll_fit_after,
            "theta_fit_raw": theta_fit_raw,
            "fit_acc_at_theta_raw": fit_acc_raw,
            "theta_fit_cal": theta_fit_cal,
            "fit_acc_at_theta_cal": fit_acc_cal,
            "theta_budget50": best_th_50,
        },
        "discrimination": {
            "auroc": make_ci_dict(auroc_pt, b_auroc, n_test),
            "auprc": make_ci_dict(auprc_pt, b_auprc, n_test),
            "log_loss_raw": make_ci_dict(nll_raw, b_nll_raw, n_test),
            "log_loss_temp": make_ci_dict(nll_cal, b_nll_cal, n_test),
            "brier_raw": make_ci_dict(brier_raw, b_brier_raw, n_test),
            "brier_temp": make_ci_dict(brier_cal, b_brier_cal, n_test),
            "apgr": make_ci_dict(pareto_main["apgr"], b_apgr, n_test),
            "pgr_at_50pct_cost": make_ci_dict(pareto_main["milestones"]["pct_50"]["pgr"], b_pgr50, n_test),
        },
        "pareto": pareto_main,
        "pareto_temp_scaled": pareto_cal,
        "regimes": regimes,
        "calibration": calib,
        "reorder": {
            "flip_rate": make_ci_dict(float(flips.mean()), flip_boot, n_test, {"flip_count": int(flips.sum())}),
            "reversed_acc_050": make_ci_dict(float(rev_acc.mean()), rev_acc_boot, n_test),
            "max_abs_pfr_shift": float(rev_pfr_diff.max()),
            "mean_abs_pfr_shift": float(rev_pfr_diff.mean()),
        },
        "per_family": per_family,
        "secondary_pairs": secondary_pairs,
        "_arrays": {
            "y_t": y_t,
            "pfr_t": pfr_t,
            "pfr_t_cal": pfr_t_cal,
            "ssm_t": ssm_t,
            "sfr_t": sfr_t,
            "csm_t": csm_t,
            "cfr_t": cfr_t,
            "auroc_boot": b_auroc,
            "apgr_boot": b_apgr,
        },
    }


def analyze_part1_cascade() -> dict | None:
    """Evaluate 2-model decision cascade (d1-2B local -> Drex Max v1 GPU) on Part 1 n=464 arXiv test papers."""
    papers_path = ROOT / "data" / "papers.jsonl"
    d1_path = ROOT / "results" / "d1_2b.jsonl"
    drex_path = ROOT / "results" / "drex_max.jsonl"
    if not (papers_path.exists() and d1_path.exists() and drex_path.exists()):
        return None

    papers = [json.loads(l) for l in papers_path.read_text().splitlines() if l.strip()]
    d1_map = {r["id"]: r for r in (json.loads(l) for l in d1_path.read_text().splitlines() if l.strip())}
    dx_map = {r["id"]: r for r in (json.loads(l) for l in drex_path.read_text().splitlines() if l.strip())}

    from common import CATEGORIES
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}

    fit_p = [p for p in papers if p["split"] == "fit"]
    test_p = [p for p in papers if p["split"] == "test"]

    def extract_mat(plist, rmap):
        probs = np.array([[rmap[p["id"]]["probabilities"][c] for c in CATEGORIES] for p in plist], dtype=float)
        y = np.array([cat_to_idx[p["primary"]] for p in plist], dtype=int)
        return probs, y

    d1_pf, y_f = extract_mat(fit_p, d1_map)
    d1_pt, y_t = extract_mat(test_p, d1_map)
    dx_pt, _ = extract_mat(test_p, dx_map)

    tau_d1, _, _ = fit_temperature(d1_pf, y_f)
    d1_pf_cal = apply_temperature(d1_pf, tau_d1)
    d1_pt_cal = apply_temperature(d1_pt, tau_d1)

    d1_ch = d1_pt.argmax(axis=1)
    dx_ch = dx_pt.argmax(axis=1)
    d1_corr = (d1_ch == y_t).astype(float)
    dx_corr = (dx_ch == y_t).astype(float)
    d1_pt_raw = d1_pt.max(axis=1)
    d1_pt_c = d1_pt_cal.max(axis=1)

    # Fit 95%-precision coverage cutoff on d1 fit temp-scaled (n=96) vs Part 1 test reported 0.859
    from analyze import coverage_at_target_precision
    cov95_fit = coverage_at_target_precision(d1_pf_cal.max(axis=1), (d1_pf_cal.argmax(axis=1) == y_f).astype(float), 0.95)
    # Also use Part 1 test-verified 0.85935 cutoff reported earlier
    thr_95_fit = cov95_fit["threshold"] if cov95_fit["threshold"] is not None else 0.859

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(test_p), size=(N_BOOT, len(test_p)))

    policies = {}
    for pname, mask_local in [
        ("all_d1", np.ones(len(test_p), dtype=bool)),
        ("d1_raw_ge_093", d1_pt_raw >= 0.93),
        ("d1_temp_ge_093", d1_pt_c >= 0.93),
        ("d1_temp_fit95prec", d1_pt_c >= thr_95_fit),
        ("d1_temp_ge_0859", d1_pt_c >= 0.859348289401325),
        ("all_drex", np.zeros(len(test_p), dtype=bool)),
    ]:
        comb = np.where(mask_local, d1_corr, dx_corr)
        local_frac = mask_local.astype(float)
        comb_boot = comb[boot_idx].mean(axis=1)
        loc_boot = local_frac[boot_idx].mean(axis=1)
        policies[pname] = {
            "local_pct": make_ci_dict(float(local_frac.mean()), loc_boot, len(test_p)),
            "combined_acc": make_ci_dict(float(comb.mean()), comb_boot, len(test_p)),
            "local_count": int(mask_local.sum()),
            "escalated_count": int((~mask_local).sum()),
        }

    # Full offload-vs-accuracy curve on temp-scaled d1 confidence
    order = np.argsort(-d1_pt_c, kind="mergesort")
    cum_loc = np.cumsum(d1_corr[order])
    tot_dx = float(dx_corr.sum())
    cum_dx_rem = tot_dx - np.cumsum(dx_corr[order])
    n = len(test_p)
    offload_curve = {
        "local_frac": [0.0] + ((np.arange(1, n + 1)) / n).tolist(),
        "combined_acc": [float(dx_corr.mean())] + ((cum_loc + cum_dx_rem) / n).tolist(),
    }

    return {
        "n_fit": len(fit_p),
        "n_test": len(test_p),
        "tau_d1": float(tau_d1),
        "fit95_threshold": float(thr_95_fit),
        "policies": policies,
        "offload_curve_temp_d1": offload_curve,
    }


def main() -> None:
    queries_path = ROOT / "data" / "router_queries.jsonl"
    d1_path = ROOT / "results" / "router_d1_2b.jsonl"
    d1_rev_path = ROOT / "results" / "router_d1_2b_reversed.jsonl"
    dx_path = ROOT / "results" / "router_drex_max.jsonl"
    dx_rev_path = ROOT / "results" / "router_drex_max_reversed.jsonl"

    queries = [json.loads(l) for l in queries_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    test_q = [q for q in queries if q["split"] == "test"]
    n_test = len(test_q)

    def load_map(p: Path) -> dict[str, dict]:
        return {r["id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx_test = rng.integers(0, n_test, size=(N_BOOT, n_test))

    drex_res = analyze_router_model("drex-max-v1", queries, load_map(dx_path), load_map(dx_rev_path), boot_idx_test)
    d1_res = analyze_router_model("d1-2B", queries, load_map(d1_path), load_map(d1_rev_path), boot_idx_test)

    # Cross-model threshold swap test (Appendix M: applying Model A's fitted threshold to Model B!)
    dx_arr = drex_res.pop("_arrays")
    d1_arr = d1_res.pop("_arrays")

    drex_using_d1_th = eval_fixed_policy(
        dx_arr["pfr_t"] >= d1_res["fit"]["theta_fit_raw"],
        dx_arr["ssm_t"], dx_arr["sfr_t"], dx_arr["csm_t"], dx_arr["cfr_t"], dx_arr["y_t"], boot_idx_test,
    )
    d1_using_drex_th = eval_fixed_policy(
        d1_arr["pfr_t"] >= drex_res["fit"]["theta_fit_raw"],
        d1_arr["ssm_t"], d1_arr["sfr_t"], d1_arr["csm_t"], d1_arr["cfr_t"], d1_arr["y_t"], boot_idx_test,
    )

    # Paired bootstrap differences (Drex - d1)
    auroc_diff = make_ci_dict(
        drex_res["discrimination"]["auroc"]["value"] - d1_res["discrimination"]["auroc"]["value"],
        dx_arr["auroc_boot"] - d1_arr["auroc_boot"],
        n_test,
    )
    apgr_diff = make_ci_dict(
        drex_res["discrimination"]["apgr"]["value"] - d1_res["discrimination"]["apgr"]["value"],
        dx_arr["apgr_boot"] - d1_arr["apgr_boot"],
        n_test,
    )

    # Prompt-length heuristic baseline & Always baselines on test
    lengths_t = np.array([float(q["prompt_chars"]) for q in test_q])
    len_pareto = compute_pareto_curve(lengths_t, dx_arr["ssm_t"], dx_arr["sfr_t"], dx_arr["csm_t"], dx_arr["cfr_t"])
    b_len_auroc = np.empty(N_BOOT, dtype=float)
    b_len_apgr = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        idx = boot_idx_test[b]
        yb = dx_arr["y_t"][idx]
        b_len_auroc[b] = roc_auc_score(yb, lengths_t[idx]) if 0 < yb.sum() < len(yb) else np.nan
        b_len_apgr[b] = compute_pareto_curve(
            lengths_t[idx], dx_arr["ssm_t"][idx], dx_arr["sfr_t"][idx], dx_arr["csm_t"][idx], dx_arr["cfr_t"][idx]
        )["apgr"]

    baselines = {
        "always_small": eval_fixed_policy(
            np.zeros(n_test, dtype=bool), dx_arr["ssm_t"], dx_arr["sfr_t"], dx_arr["csm_t"], dx_arr["cfr_t"], dx_arr["y_t"], boot_idx_test
        ),
        "always_frontier": eval_fixed_policy(
            np.ones(n_test, dtype=bool), dx_arr["ssm_t"], dx_arr["sfr_t"], dx_arr["csm_t"], dx_arr["cfr_t"], dx_arr["y_t"], boot_idx_test
        ),
        "oracle": eval_fixed_policy(
            dx_arr["y_t"].astype(bool), dx_arr["ssm_t"], dx_arr["sfr_t"], dx_arr["csm_t"], dx_arr["cfr_t"], dx_arr["y_t"], boot_idx_test
        ),
        "prompt_length_heuristic": {
            "auroc": make_ci_dict(float(roc_auc_score(dx_arr["y_t"], lengths_t)), b_len_auroc, n_test),
            "apgr": make_ci_dict(len_pareto["apgr"], b_len_apgr, n_test),
            "pareto": len_pareto,
        },
    }

    cascade = analyze_part1_cascade()

    summary_out = {
        "config": {
            "n_fit": 100,
            "n_test": n_test,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "primary_small": PRIMARY_SMALL_MODEL,
            "primary_frontier": PRIMARY_FRONTIER_MODEL,
        },
        "baselines": baselines,
        "models": {
            "drex_max_v1": drex_res,
            "d1_2b": d1_res,
        },
        "comparisons": {
            "auroc_diff_drex_minus_d1": auroc_diff,
            "apgr_diff_drex_minus_d1": apgr_diff,
            "cross_threshold_swap": {
                "drex_using_d1_theta": drex_using_d1_th,
                "d1_using_drex_theta": d1_using_drex_th,
            },
        },
        "part1_decision_cascade": cascade,
    }

    out_json = ROOT / "results" / "router_metrics.json"
    out_json.write_text(json.dumps(summary_out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}", flush=True)

    # Human-readable Markdown summary
    def fmt_pct(d: dict) -> str:
        return f"{d['value']*100:.1f}% [{d['ci95'][0]*100:.1f}, {d['ci95'][1]*100:.1f}]"

    def fmt_flt(d: dict, digits: int = 3) -> str:
        return f"{d['value']:.{digits}f} [{d['ci95'][0]:.{digits}f}, {d['ci95'][1]:.{digits}f}]"

    lines = [
        "# Part 2 LLM-Router Summary (`CARROT-LLM-Routing/SPROUT-o3mini`, n=500 test, 100 fit)",
        "",
        "## Baselines (Llama-3.2-3B -> OpenAI o3-mini)",
        f"- Always Small (`Llama-3.2-3B`): Quality {fmt_pct(baselines['always_small']['system_quality'])}, Cost ${baselines['always_small']['cost_per_1k']['value']:.3f}/1k",
        f"- Always Frontier (`o3-mini`): Quality {fmt_pct(baselines['always_frontier']['system_quality'])}, Cost ${baselines['always_frontier']['cost_per_1k']['value']:.3f}/1k",
        f"- Oracle Upper Bound: Quality {fmt_pct(baselines['oracle']['system_quality'])}, Cost ${baselines['oracle']['cost_per_1k']['value']:.3f}/1k, Route Acc {fmt_pct(baselines['oracle']['routing_accuracy'])}",
        f"- Prompt-Length Heuristic: AUROC {fmt_flt(baselines['prompt_length_heuristic']['auroc'])}, APGR {fmt_flt(baselines['prompt_length_heuristic']['apgr'])}",
        "",
        "## Model Comparison (n=500 test)",
        f"- Drex Max v1: AUROC {fmt_flt(drex_res['discrimination']['auroc'])}, APGR {fmt_flt(drex_res['discrimination']['apgr'])}, PGR@50%Cost {fmt_pct(drex_res['discrimination']['pgr_at_50pct_cost'])}, tau={drex_res['fit']['tau']:.3f}, theta_raw={drex_res['fit']['theta_fit_raw']:.3f}",
        f"- d1-2B:       AUROC {fmt_flt(d1_res['discrimination']['auroc'])}, APGR {fmt_flt(d1_res['discrimination']['apgr'])}, PGR@50%Cost {fmt_pct(d1_res['discrimination']['pgr_at_50pct_cost'])}, tau={d1_res['fit']['tau']:.3f}, theta_raw={d1_res['fit']['theta_fit_raw']:.3f}",
        f"- Delta (Drex - d1): AUROC {fmt_flt(auroc_diff)}, APGR {fmt_flt(apgr_diff)}",
        "",
        "## Threshold Transfer & 100-Label Tuning (Routing Accuracy / Cost / Quality)",
    ]
    for mlabel, mres in [("Drex Max v1", drex_res), ("d1-2B", d1_res)]:
        for rname in ("argmax_050", "nace_080", "fit100_raw_theta", "fit100_temp_theta"):
            rg = mres["regimes"][rname]
            lines.append(
                f"- {mlabel} | {rname:18s}: RouteAcc={fmt_pct(rg['routing_accuracy'])}, Esc={rg['esc_rate']['value']*100:.1f}%, SysQual={rg['system_quality']['value']*100:.1f}%, Cost=${rg['cost_per_1k']['value']:.2f}/1k"
            )
    lines.append(
        f"- Cross-swap: Drex@d1_theta RouteAcc={fmt_pct(drex_using_d1_th['routing_accuracy'])}, d1@Drex_theta RouteAcc={fmt_pct(d1_using_drex_th['routing_accuracy'])}"
    )

    if cascade:
        lines.extend(["", "## Part 1 Decision-Model Cascade (`d1-2B` Local -> `Drex Max v1` GPU, n=464)"])
        for pname, pd in cascade["policies"].items():
            lines.append(
                f"- {pname:18s}: Local={pd['local_pct']['value']*100:.1f}% ({pd['local_count']}/464), Combined Acc={fmt_pct(pd['combined_acc'])}"
            )

    out_md = ROOT / "results" / "router_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
