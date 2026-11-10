#!/usr/bin/env python3
"""Part 2 — Generate plain-English charts for the LLM-Router experiment & Part 1 decision cascade."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

BG_COLOR = "#FAF9F6"
PANEL_BG = "#FFFFFF"
TEXT_DARK = "#111827"
TEXT_MUTED = "#4B5563"
GRID_COLOR = "#E5E7EB"

DREX_COLOR = "#1D4ED8"
DREX_AFTER = "#0F766E"
D1_COLOR = "#D97706"
D1_AFTER = "#7C3AED"
REF_RED = "#DC2626"
ORACLE_GREEN = "#15803D"
LENGTH_GRAY = "#374151"


def setup_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#D1D5DB",
            "axes.linewidth": 0.9,
            "axes.facecolor": PANEL_BG,
            "figure.facecolor": BG_COLOR,
            "text.color": TEXT_DARK,
            "axes.labelcolor": TEXT_DARK,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
        }
    )


def plot_router_pareto(data: dict, out_path: Path) -> None:
    base = data["baselines"]
    dx = data["models"]["drex_max_v1"]
    d1 = data["models"]["d1_2b"]
    n_test = data["config"]["n_test"]

    c_sm = base["always_small"]["cost_per_1k"]["value"]
    q_sm = base["always_small"]["system_quality"]["value"] * 100.0
    c_fr = base["always_frontier"]["cost_per_1k"]["value"]
    q_fr = base["always_frontier"]["system_quality"]["value"] * 100.0
    c_or = base["oracle"]["cost_per_1k"]["value"]
    q_or = base["oracle"]["system_quality"]["value"] * 100.0

    fig, ax = plt.subplots(figsize=(9.8, 6.2), dpi=160, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.14)

    # Random chord
    ax.plot([c_sm, c_fr], [q_sm, q_fr], linestyle="--", linewidth=1.6, color="#9CA3AF", label="Random routing baseline (50% APGR)", zorder=2)

    # Prompt-length heuristic
    lp = base["prompt_length_heuristic"]["pareto"]
    ax.plot(
        lp["cost_per_1k"],
        [q * 100.0 for q in lp["quality"]],
        linestyle=":",
        linewidth=2.0,
        color=LENGTH_GRAY,
        label=f"Prompt-length heuristic (APGR {base['prompt_length_heuristic']['apgr']['value']*100:.1f}%)",
        zorder=3,
    )

    # d1-2B & Drex Max curves
    d1_p = d1["pareto"]
    dx_p = dx["pareto"]
    ax.plot(
        d1_p["cost_per_1k"],
        [q * 100.0 for q in d1_p["quality"]],
        linewidth=2.6,
        color=D1_COLOR,
        label=f"d1-2B (1.9B decision model) · APGR {dx_or_pct(d1['discrimination']['apgr']['value'])}",
        zorder=4,
    )
    ax.plot(
        dx_p["cost_per_1k"],
        [q * 100.0 for q in dx_p["quality"]],
        linewidth=2.8,
        color=DREX_COLOR,
        label=f"Drex Max v1 (27B decision model) · APGR {dx_or_pct(dx['discrimination']['apgr']['value'])}",
        zorder=5,
    )

    # Endpoints & Oracle
    ax.scatter([c_sm], [q_sm], s=85, color="#374151", zorder=6)
    ax.scatter([c_fr], [q_fr], s=85, color="#111827", zorder=6)
    ax.scatter([c_or], [q_or], s=110, marker="*", color=ORACLE_GREEN, zorder=7, label=f"Oracle upper bound ({q_or:.1f}% at ${c_or:.2f}/1k)")

    ax.annotate(
        f"Always 3B\n{q_sm:.1f}%, ${c_sm:.2f}/1k",
        xy=(c_sm, q_sm),
        xytext=(c_sm + 0.16, q_sm - 2.6),
        fontsize=8.5,
        fontweight="bold",
        color="#374151",
    )
    ax.annotate(
        f"Always o3-mini\n{q_fr:.1f}%, ${c_fr:.2f}/1k",
        xy=(c_fr, q_fr),
        xytext=(c_fr - 1.35, q_fr + 1.0),
        fontsize=8.5,
        fontweight="bold",
        color="#111827",
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.0, shrinkA=2, shrinkB=5),
    )

    # 50% cost vertical guide + milestone dots
    c_half = 0.5 * c_fr
    ax.axvline(c_half, color=REF_RED, linestyle="-.", linewidth=1.1, alpha=0.6)
    ax.text(c_half - 0.07, q_fr - 4.2, f"50% cost budget\n(${c_half:.2f}/1k)", fontsize=8.3, color=REF_RED, ha="right", va="top")
    dx_50 = dx_p["milestones"]["pct_50"]["quality"] * 100.0
    d1_50 = d1_p["milestones"]["pct_50"]["quality"] * 100.0
    rnd_50 = dx_p["milestones"]["pct_50"]["random_quality"] * 100.0
    ax.scatter([c_half, c_half, c_half], [dx_50, d1_50, rnd_50], s=[65, 65, 45], color=[DREX_COLOR, D1_COLOR, "#9CA3AF"], edgecolor="white", linewidth=1.0, zorder=8)

    ax.set_xlim(-0.15, c_fr + 0.35)
    ax.set_ylim(max(55.0, q_sm - 5.0), min(100.0, max(q_fr, q_or) + 3.8))
    ax.set_xlabel("Average LLM Inference Cost per 1,000 Prompts ($ USD)")
    ax.set_ylabel("Mean Downstream Answer Quality (%)")
    ax.legend(loc="lower right", frameon=True, facecolor=PANEL_BG, framealpha=0.95, edgecolor="#E5E7EB", fontsize=8.7)

    fig.text(0.09, 0.94, "Cost vs. Answer Quality: Zero-Shot LLM Routing (3B vs. o3-mini)", fontsize=14, fontweight="bold")
    fig.text(
        0.09,
        0.895,
        f"SPROUT-o3mini held-out test split (n = {n_test}); sweeping probability threshold p(needs frontier) from 0 to 1",
        fontsize=9.5,
        color=TEXT_MUTED,
    )
    fig.text(
        0.09,
        0.03,
        "Data: CARROT-LLM-Routing/SPROUT-o3mini (Llama-3.2-3B $0.06/1M vs. OpenAI o3-mini $1.10/$4.40 per 1M). APGR = normalized area recovery.",
        fontsize=8.0,
        color=TEXT_MUTED,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def dx_or_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def plot_router_thresholds(data: dict, out_path: Path) -> None:
    dx = data["models"]["drex_max_v1"]
    d1 = data["models"]["d1_2b"]
    swap = data["comparisons"]["cross_threshold_swap"]
    n_test = data["config"]["n_test"]

    labels = [
        "Default 50%\n(cutoff 0.50)",
        "Nace §M cutoff\n(cutoff 0.80)",
        "Cross-model\nswap (unfit)",
        "100-label tuned\n(on fit split)",
    ]
    dx_regs = [
        dx["regimes"]["argmax_050"]["routing_accuracy"],
        dx["regimes"]["nace_080"]["routing_accuracy"],
        swap["drex_using_d1_theta"]["routing_accuracy"],
        dx["regimes"]["fit100_temp_theta"]["routing_accuracy"],
    ]
    d1_regs = [
        d1["regimes"]["argmax_050"]["routing_accuracy"],
        d1["regimes"]["nace_080"]["routing_accuracy"],
        swap["d1_using_drex_th"]["routing_accuracy"] if "d1_using_drex_th" in swap else swap["d1_using_drex_theta"]["routing_accuracy"],
        d1["regimes"]["fit100_temp_theta"]["routing_accuracy"],
    ]

    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=160, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.82, bottom=0.17)

    dx_vals = [r["value"] * 100.0 for r in dx_regs]
    dx_his = [r["ci95"][1] * 100.0 for r in dx_regs]
    dx_err = [
        [(r["value"] - r["ci95"][0]) * 100.0 for r in dx_regs],
        [(r["ci95"][1] - r["value"]) * 100.0 for r in dx_regs],
    ]
    d1_vals = [r["value"] * 100.0 for r in d1_regs]
    d1_his = [r["ci95"][1] * 100.0 for r in d1_regs]
    d1_err = [
        [(r["value"] - r["ci95"][0]) * 100.0 for r in d1_regs],
        [(r["ci95"][1] - r["value"]) * 100.0 for r in d1_regs],
    ]

    b1 = ax.bar(
        x - w / 2,
        dx_vals,
        width=w,
        yerr=dx_err,
        capsize=4,
        color=DREX_COLOR,
        alpha=0.9,
        label=f"Drex Max v1 (fit100 θ_cal={dx['fit']['theta_fit_cal']:.2f} / θ_raw={dx['fit']['theta_fit_raw']:.2f}, τ={dx['fit']['tau']:.2f})",
        zorder=3,
    )
    b2 = ax.bar(
        x + w / 2,
        d1_vals,
        width=w,
        yerr=d1_err,
        capsize=4,
        color=D1_COLOR,
        alpha=0.9,
        label=f"d1-2B (fit100 θ_cal={d1['fit']['theta_fit_cal']:.2f} / θ_raw={d1['fit']['theta_fit_raw']:.2f}, τ={d1['fit']['tau']:.2f})",
        zorder=3,
    )

    ax.axhline(50.8, color="#6B7280", linestyle="--", linewidth=1.2, label="Always pick majority class (50.8%)", zorder=2)

    for rect, val, hi in zip(b1, dx_vals, dx_his):
        ax.text(rect.get_x() + rect.get_width() / 2, hi + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=DREX_COLOR)
    for rect, val, hi in zip(b2, d1_vals, d1_his):
        ax.text(rect.get_x() + rect.get_width() / 2, hi + 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#92400E")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylim(35, 82)
    ax.set_ylabel("Routing Decision Accuracy (%)")
    ax.legend(loc="upper left", frameon=True, facecolor=PANEL_BG, framealpha=0.95, edgecolor="#E5E7EB", fontsize=8.4)

    fig.text(0.09, 0.93, "Fixed Cutoffs Don't Transfer: How 100 Labeled Queries Fix Thresholds", fontsize=13.6, fontweight="bold")
    fig.text(
        0.09,
        0.885,
        f"Exact routing accuracy (95% bootstrap CIs, n = {n_test}) testing default 0.50, Nace §M 0.80, cross-model swap, and tuned cutoffs",
        fontsize=9.3,
        color=TEXT_MUTED,
    )
    fig.text(
        0.09,
        0.03,
        "Ref: Nace arXiv:2611.07854 §M warns thresholds shift across models and must be re-fitted whenever the model changes.",
        fontsize=8.0,
        color=TEXT_MUTED,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_router_calibration_and_families(data: dict, out_path: Path) -> None:
    dx = data["models"]["drex_max_v1"]
    d1 = data["models"]["d1_2b"]
    n_test = data["config"]["n_test"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.7), dpi=160, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.17, wspace=0.25)

    # Panel 1: Calibration curve (p_top before & after temp scaling)
    ax1.plot([50, 100], [50, 100], linestyle="--", linewidth=1.3, color="#6B7280", label="Perfect match", zorder=2)

    def pts(bins: list[dict], min_n: int = 3):
        xs, ys = [], []
        for b in bins:
            if b["count"] >= min_n and b["mean_conf"] is not None:
                xs.append(b["mean_conf"] * 100.0)
                ys.append(b["accuracy"] * 100.0)
        return xs, ys

    for bins, col, ls, mk, lbl in [
        (dx["calibration"]["raw_p_top"]["bins"], DREX_COLOR, "--", "o", f"Drex raw (ECE {dx['calibration']['raw_p_top']['ece']['value']*100:.1f}%)"),
        (dx["calibration"]["temp_p_top"]["bins"], DREX_AFTER, "-", "s", f"Drex + 100-label τ (ECE {dx['calibration']['temp_p_top']['ece']['value']*100:.1f}%)"),
        (d1["calibration"]["raw_p_top"]["bins"], D1_COLOR, "--", "^", f"d1 raw (ECE {d1['calibration']['raw_p_top']['ece']['value']*100:.1f}%)"),
        (d1["calibration"]["temp_p_top"]["bins"], D1_AFTER, "-", "D", f"d1 + 100-label τ (ECE {d1['calibration']['temp_p_top']['ece']['value']*100:.1f}%)"),
    ]:
        xs, ys = pts(bins)
        ax1.plot(xs, ys, linestyle=ls, marker=mk, markersize=5.5, linewidth=2.0, color=col, label=lbl, zorder=4)

    ax1.axvline(93, color=REF_RED, linestyle=":", linewidth=1.2, alpha=0.75)
    ax1.set_xlim(48, 101)
    ax1.set_ylim(25, 102)
    ax1.set_xlabel("Model's Stated Top Probability (%)")
    ax1.set_ylabel("Actual Binary Routing Accuracy (%)")
    ax1.set_title("Confidence Calibration on LLM Routing", fontsize=10.8, fontweight="bold", pad=8)
    ax1.legend(loc="lower right", frameon=True, facecolor=PANEL_BG, edgecolor="#E5E7EB", fontsize=7.6)

    # Panel 2: Per-family routing escalation rates (Truth vs Drex vs d1)
    fams = sorted(dx["per_family"].keys(), key=lambda f: -dx["per_family"][f]["n"])
    fam_labels = [f"{f}\n(n={dx['per_family'][f]['n']})" for f in fams]
    gold_esc = [dx["per_family"][f]["gold_frontier_frac"] * 100.0 for f in fams]
    dx_esc = [dx["per_family"][f]["pred_frontier_frac_tuned"] * 100.0 for f in fams]
    d1_esc = [d1["per_family"][f]["pred_frontier_frac_tuned"] * 100.0 for f in fams]

    x2 = np.arange(len(fams))
    w2 = 0.26
    ax2.bar(x2 - w2, gold_esc, width=w2, color="#374151", alpha=0.85, label="Actual frontier needed (3B < o3-mini)", zorder=3)
    b_dx = ax2.bar(x2, dx_esc, width=w2, color=DREX_COLOR, alpha=0.9, label="Drex Max v1 escalates (tuned θ)", zorder=3)
    b_d1 = ax2.bar(x2 + w2, d1_esc, width=w2, color=D1_COLOR, alpha=0.9, label="d1-2B escalates (tuned θ)", zorder=3)

    for rect, val in zip(b_dx, dx_esc):
        if val < 1.0:
            ax2.text(rect.get_x() + rect.get_width() / 2, 1.5, "0%", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=DREX_COLOR)
    for rect, val in zip(b_d1, d1_esc):
        if val < 1.0:
            ax2.text(rect.get_x() + rect.get_width() / 2, 1.5, "0%", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#92400E")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(fam_labels, fontsize=8.3)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Share Routed to Frontier o3-mini (%)")
    ax2.set_title("Where Escalation Happens by Benchmark Type", fontsize=10.8, fontweight="bold", pad=8)
    ax2.legend(loc="upper left", frameon=True, facecolor=PANEL_BG, edgecolor="#E5E7EB", fontsize=7.6)

    fig.text(0.07, 0.93, "Routing Calibration and Per-Benchmark Behavior", fontsize=14, fontweight="bold")
    fig.text(
        0.07,
        0.885,
        f"Left: reliability curves before/after 100-label temperature fix (n = {n_test}). Right: actual vs. predicted o3-mini escalation rates.",
        fontsize=9.3,
        color=TEXT_MUTED,
    )
    fig.text(
        0.07,
        0.03,
        "Zero-shot binary routing (`small` vs `frontier`) on CARROT-LLM-Routing/SPROUT-o3mini test queries.",
        fontsize=8.0,
        color=TEXT_MUTED,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_part1_cascade(data: dict, out_path: Path) -> None:
    cas = data.get("part1_decision_cascade")
    if not cas:
        return
    curve = cas["offload_curve_temp_d1"]
    pol = cas["policies"]

    fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=160, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.83, bottom=0.14)

    xs = [v * 100.0 for v in curve["local_frac"]]
    ys = [v * 100.0 for v in curve["combined_acc"]]
    ax.plot(xs, ys, linewidth=2.6, color=DREX_AFTER, label="Temp-scaled d1-2B → Drex Max v1 cascade", zorder=4)

    drex_acc = pol["all_drex"]["combined_acc"]["value"] * 100.0
    d1_acc = pol["all_d1"]["combined_acc"]["value"] * 100.0
    ax.axhline(drex_acc, linestyle="--", linewidth=1.2, color=DREX_COLOR, alpha=0.7, label=f"All Drex Max v1 on A100 GPU ({drex_acc:.1f}%)")
    ax.axhline(d1_acc, linestyle=":", linewidth=1.2, color=D1_COLOR, alpha=0.85, label=f"All d1-2B on laptop M3 ({d1_acc:.1f}%)")

    pts = [
        ("Raw ≥ 93%\n(4.5% local)", pol["d1_raw_ge_093"], D1_COLOR, (7, -1.9)),
        ("96-label τ fix ≥ 93%\n(18.8% local, zero drop)", pol["d1_temp_ge_093"], DREX_COLOR, (20, 0.9)),
        ("Fit-96 95%-prec (≥ 90.2%)\n(25.4% local, -0.2 pt)", pol["d1_temp_fit95prec"], D1_AFTER, (27, -3.2)),
    ]
    for txt, pd, col, (tx, dy) in pts:
        px = pd["local_pct"]["value"] * 100.0
        py = pd["combined_acc"]["value"] * 100.0
        ax.scatter([px], [py], s=80, color=col, edgecolor="white", linewidth=1.2, zorder=6)
        ax.annotate(f"{txt}\n({py:.1f}% acc)", xy=(px, py), xytext=(tx, py + dy), fontsize=8.3, fontweight="bold", color=col, arrowprops=dict(arrowstyle="->", color=col, lw=1.0))

    ax.set_xlim(-2, 102)
    ax.set_ylim(min(d1_acc, min(ys)) - 2.5, max(drex_acc, max(ys)) + 2.5)
    ax.set_xlabel("Share of Unseen arXiv Papers Kept on Local d1-2B (1.9B) (%)")
    ax.set_ylabel("Combined 8-Category Accuracy (%)")
    ax.legend(loc="lower left", frameon=True, facecolor=PANEL_BG, edgecolor="#E5E7EB", fontsize=8.6)

    fig.text(0.09, 0.935, "Part 1 Model-Selection Scenario: Local d1-2B → Drex Max v1 Cascade", fontsize=13.8, fontweight="bold")
    fig.text(
        0.09,
        0.89,
        f"Unseen Nov 1–8, 2026 arXiv papers (n = {cas['n_test']}). Escalate uncertain queries to 27B Drex Max on A100 GPU.",
        fontsize=9.3,
        color=TEXT_MUTED,
    )
    fig.text(
        0.09,
        0.03,
        "96-label temperature fix on d1-2B (τ = 0.701) lifts zero-loss local offload from 4.5% (21/464) to 18.8% (87/464).",
        fontsize=8.0,
        color=TEXT_MUTED,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    setup_rc()
    metrics_path = ROOT / "results" / "router_metrics.json"
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    plot_router_pareto(data, assets_dir / "router_pareto.png")
    plot_router_thresholds(data, assets_dir / "router_thresholds.png")
    plot_router_calibration_and_families(data, assets_dir / "router_calibration_and_families.png")
    plot_part1_cascade(data, assets_dir / "cascade_arxiv.png")
    print("Generated 4 Part-2 charts in assets/")


if __name__ == "__main__":
    main()
