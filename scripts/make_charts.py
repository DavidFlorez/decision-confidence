#!/usr/bin/env python3
"""Step 6 — Generate plain-English, publication-grade PNG charts under assets/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent

# Palette: colorblind-friendly editorial tones
BG_COLOR = "#FAF9F6"
PANEL_BG = "#FFFFFF"
TEXT_DARK = "#111827"
TEXT_MUTED = "#4B5563"
GRID_COLOR = "#E5E7EB"

DREX_COLOR = "#1D4ED8"       # Deep royal blue
DREX_AFTER = "#0F766E"       # Deep teal (after temp fix)
D1_COLOR = "#D97706"         # Warm amber-ochre
D1_AFTER = "#7C3AED"         # Crisp violet (after temp fix)
REF_RED = "#DC2626"          # Benchmark reference line

SHORT_LABELS = {
    "cs.CL": "Language\n(cs.CL)",
    "cs.CV": "Vision\n(cs.CV)",
    "cs.LG": "Machine\nLearning\n(cs.LG)",
    "cs.RO": "Robotics\n(cs.RO)",
    "cs.CR": "Security\n(cs.CR)",
    "cs.SE": "Software\nEng.\n(cs.SE)",
    "cs.DC": "Distributed\n(cs.DC)",
    "cs.HC": "Human-\nComp.\n(cs.HC)",
}

ROW_LABELS = {
    "cs.CL": "Language (cs.CL)",
    "cs.CV": "Vision (cs.CV)",
    "cs.LG": "Machine Learning (cs.LG)",
    "cs.RO": "Robotics (cs.RO)",
    "cs.CR": "Security (cs.CR)",
    "cs.SE": "Software Eng. (cs.SE)",
    "cs.DC": "Distributed (cs.DC)",
    "cs.HC": "Human-Comp. (cs.HC)",
}


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
            "grid.alpha": 0.8,
        }
    )


def plot_reliability(model_data: dict, short_title_name: str, before_color: str, after_color: str, out_path: Path) -> None:
    cal = model_data["calibration"]
    ts = model_data["temperature_scaling"]
    bins_before = cal["reliability_bins_p_top_before"]
    bins_after = cal["reliability_bins_p_top_after"]

    fig = plt.figure(figsize=(9.5, 7.6), dpi=160, facecolor=BG_COLOR)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.15], hspace=0.16, left=0.10, right=0.96, top=0.86, bottom=0.09)
    ax = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax)

    # Perfect calibration diagonal
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.5, color="#6B7280", zorder=2, label="Perfect match (stated % = actual % right)")

    def extract_pts(bins: list[dict], min_n: int = 2) -> tuple[list[float], list[float], list[int]]:
        xs, ys, ns = [], [], []
        for b in bins:
            if b["count"] >= min_n and b["mean_conf"] is not None and b["accuracy"] is not None:
                xs.append(b["mean_conf"] * 100.0)
                ys.append(b["accuracy"] * 100.0)
                ns.append(b["count"])
        return xs, ys, ns

    xb, yb, nb = extract_pts(bins_before, min_n=2)
    xa, ya, na = extract_pts(bins_after, min_n=2)

    ece_b = cal["ece_15bin_p_top_before"]["value"] * 100.0
    ece_a = cal["ece_15bin_p_top_after"]["value"] * 100.0
    tau_v = ts["fitted_multiplier_tau"]["value"]

    ax.plot(
        xb, yb, "-o", color=before_color, linewidth=2.4, markersize=7.5, zorder=4,
        label=f"Out-of-the-box (avg gap from diagonal: {ece_b:.1f} percentage points)",
    )
    ax.plot(
        xa, ya, "-s", color=after_color, linewidth=2.4, markersize=7.0, alpha=0.92, zorder=5,
        label=f"After 96-paper temperature fix (factor {tau_v:.2f}x; avg gap: {ece_a:.1f} pts)",
    )

    ax.set_xlim(0, 102)
    ax.set_ylim(0, 104)
    ax.set_ylabel("How often the model is right (%)", fontsize=11.5, fontweight="bold")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", frameon=True, facecolor=PANEL_BG, edgecolor="#D1D5DB", fontsize=9.5)

    # Bottom histogram of bin counts
    centers = np.array([(b["low"] + b["high"]) * 50.0 for b in bins_before])
    width = (100.0 / len(bins_before)) * 0.40
    counts_b = np.array([b["count"] for b in bins_before])
    counts_a = np.array([b["count"] for b in bins_after])

    bars_b = ax_hist.bar(centers - width / 2, counts_b, width=width, color=before_color, alpha=0.82, label="Out-of-the-box")
    bars_a = ax_hist.bar(centers + width / 2, counts_a, width=width, color=after_color, alpha=0.82, label="After temperature fix")
    ymax = max(int(counts_b.max()), int(counts_a.max()), 10)
    ax_hist.set_ylim(0, ymax * 1.28)
    for bar, cnt, col in list(zip(bars_b, counts_b, [before_color] * len(bars_b))) + list(zip(bars_a, counts_a, [after_color] * len(bars_a))):
        if cnt >= 5:
            ax_hist.text(
                bar.get_x() + bar.get_width() / 2,
                cnt + ymax * 0.025,
                f"{int(cnt)}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.4,
                fontweight="bold",
                color=col,
            )
    ax_hist.set_ylabel("Papers in bin", fontsize=10.5, fontweight="bold")
    ax_hist.set_xlabel("When the model says it is X% sure (top-choice probability, %)", fontsize=11.5, fontweight="bold")
    ax_hist.set_xlim(0, 102)
    ax_hist.legend(loc="upper left", frameon=True, facecolor=PANEL_BG, edgecolor="#D1D5DB", fontsize=8.8, ncol=2)

    fig.suptitle(
        f"When {short_title_name} says X% sure, how often is it right?",
        fontsize=15.5,
        fontweight="bold",
        x=0.10,
        y=0.955,
        ha="left",
    )
    fig.text(
        0.10,
        0.912,
        "464 unseen arXiv papers (Nov 1-8, 2026) across 8 computer science categories; bins with 2+ papers shown",
        fontsize=10.2,
        color=TEXT_MUTED,
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_claim_93(metrics: dict, out_path: Path) -> None:
    drex = metrics["models"]["drex_max"]
    d1 = metrics["models"]["d1_2b"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.5), dpi=160, sharey=True, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.81, bottom=0.16, wspace=0.10)

    regimes = [
        ("before_temp", "Out-of-the-box (checkpoint defaults)"),
        ("after_temp", "After 96-paper temperature fix"),
    ]
    def_keys = [
        ("p_top", "Raw top probability\n>= 93%"),
        ("chance_corrected", "Chance-adjusted\nconfidence >= 93%"),
    ]

    for ax, (reg_key, reg_title) in zip(axes, regimes):
        x = np.arange(len(def_keys))
        bar_w = 0.37

        for offset, mdata, mlabel, color in [
            (-bar_w / 2 - 0.02, drex, "Drex Max v1 (27B)", DREX_COLOR),
            (bar_w / 2 + 0.02, d1, "d1-2B (1.9B)", D1_COLOR),
        ]:
            vals, err_lo, err_hi, callouts = [], [], [], []
            for dkey, _ in def_keys:
                c = mdata["claim_93"][reg_key][dkey]
                acc_obj = c["accuracy_at_threshold"]
                share_val = c["share_clearing"]["value"] * 100.0
                m_cnt = c["count_clearing"]
                k_cnt = c["correct_clearing"]
                if acc_obj["value"] is None or m_cnt == 0:
                    vals.append(0.0)
                    err_lo.append(0.0)
                    err_hi.append(0.0)
                    callouts.append("0 papers\ncleared 93%")
                else:
                    v = acc_obj["value"] * 100.0
                    lo = acc_obj["ci95"][0] * 100.0
                    hi = acc_obj["ci95"][1] * 100.0
                    cp = acc_obj.get("clopper_pearson_ci95", [None, None])
                    if k_cnt == m_cnt and cp[0] is not None:
                        lo = cp[0] * 100.0
                        hi = cp[1] * 100.0
                    vals.append(v)
                    err_lo.append(max(0.0, v - lo))
                    err_hi.append(max(0.0, hi - v))
                    callouts.append(f"{v:.1f}% right\n{k_cnt}/{m_cnt} papers\n({share_val:.0f}% of set)")

            bars = ax.bar(
                x + offset,
                vals,
                width=bar_w,
                color=color,
                alpha=0.88,
                edgecolor="none",
                label=mlabel,
                zorder=3,
            )
            ax.errorbar(
                x + offset,
                vals,
                yerr=[err_lo, err_hi],
                fmt="none",
                ecolor=TEXT_DARK,
                elinewidth=1.6,
                capsize=4.5,
                zorder=5,
            )

            for bx, v, lo_e, hi_e, txt in zip(x + offset, vals, err_lo, err_hi, callouts):
                unoccluded_top = v - lo_e
                if unoccluded_top >= 36.0:
                    y_pos = unoccluded_top * 0.50
                    txt_col = "#FFFFFF"
                    va_mode = "center"
                else:
                    y_pos = v + hi_e + 2.5
                    txt_col = TEXT_DARK
                    va_mode = "bottom"
                ax.text(
                    bx,
                    y_pos,
                    txt,
                    ha="center",
                    va=va_mode,
                    fontsize=8.2,
                    fontweight="bold",
                    linespacing=1.22,
                    color=txt_col,
                    zorder=6,
                )

        ax.axhline(97.0, color=REF_RED, linestyle="--", linewidth=1.8, zorder=4, label="97% benchmark claim (Nace report)")
        ax.set_title(reg_title, fontsize=12.0, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in def_keys], fontsize=10.2, fontweight="bold")
        ax.set_ylim(0, 107)

    axes[0].set_ylabel("How often papers above 93% are right (%)", fontsize=11.5, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=True,
        facecolor=PANEL_BG,
        edgecolor="#D1D5DB",
        fontsize=10.0,
    )

    fig.suptitle(
        "At 93%+ stated confidence, how often is each model actually right?",
        fontsize=15.5,
        fontweight="bold",
        x=0.075,
        y=0.95,
        ha="left",
    )
    fig.text(
        0.075,
        0.898,
        "464 unseen arXiv test papers (Nov 1-8, 2026); error bars show 95% bootstrap CIs (exact binomial CI when 100% right)",
        fontsize=10.2,
        color=TEXT_MUTED,
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_coverage_precision(metrics: dict, out_path: Path) -> None:
    drex = metrics["models"]["drex_max"]
    d1 = metrics["models"]["d1_2b"]

    fig, ax = plt.subplots(figsize=(10.2, 6.4), dpi=160, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.095, right=0.96, top=0.83, bottom=0.13)

    # Start plotting at k >= 10 (~2.2% of 464) to avoid noisy 1-paper step jitter at far left
    min_k = 10

    for mdata, label, color in [
        (drex, "Drex Max v1 (27B)", DREX_COLOR),
        (d1, "d1-2B (1.9B)", D1_COLOR),
    ]:
        curve = mdata["coverage_at_95_precision"]["curve_primary"]
        cov = np.array(curve["coverage"]) * 100.0
        prec = np.array(curve["precision"]) * 100.0
        ks = np.array(curve["k"])
        mask = ks >= min_k

        ax.plot(cov[mask], prec[mask], color=color, linewidth=2.6, label=f"{label} — strict primary category", zorder=4)

        curve_any = mdata["coverage_at_95_precision"]["curve_any_listed"]
        prec_any = np.array(curve_any["precision"]) * 100.0
        ax.plot(
            cov[mask],
            prec_any[mask],
            color=color,
            linewidth=1.5,
            linestyle=":",
            alpha=0.75,
            label=f"{label} — counting cross-listed category too",
            zorder=3,
        )

        c95 = mdata["coverage_at_95_precision"]["coverage"]["value"] * 100.0
        if c95 > 0:
            ax.scatter([c95], [95.0], color=color, s=75, zorder=6, edgecolor="white", linewidth=1.5)
            ax.vlines(c95, 50, 95.0, colors=color, linestyles="--", linewidth=1.2, alpha=0.65)
            if mdata["model_key"] == "drex_max":
                xy_txt = (min(c95 + 4.0, 76.0), 96.3)
            else:
                xy_txt = (11.5, 85.5)
            ax.annotate(
                f"{label.split()[0]}: auto-sort {c95:.1f}%\nat 95% right",
                xy=(c95, 95.0),
                xytext=xy_txt,
                fontsize=9.2,
                fontweight="bold",
                color=color,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2},
            )

    ax.axhline(95.0, color=REF_RED, linestyle="--", linewidth=1.6, label="95% accuracy target for auto-sorting", zorder=2)

    # dynamically choose clean lower y-limit
    drex_acc = drex["classification"]["primary_accuracy"]["value"] * 100.0
    d1_acc = d1["classification"]["primary_accuracy"]["value"] * 100.0
    y_min = max(45.0, math.floor(min(drex_acc, d1_acc) / 5.0) * 5.0 - 8.0)

    ax.set_xlim(0, 102)
    ax.set_ylim(y_min, 102)
    ax.set_xlabel("Share of papers you could auto-sort, highest confidence first (%)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("How often those auto-sorted papers are right (%)", fontsize=11.5, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor=PANEL_BG, edgecolor="#D1D5DB", fontsize=9.2)

    fig.suptitle(
        "Share of papers you could auto-sort vs. how often those are right",
        fontsize=15.5,
        fontweight="bold",
        x=0.095,
        y=0.945,
        ha="left",
    )
    fig.text(
        0.095,
        0.892,
        "464 unseen arXiv test papers (Nov 1-8, 2026) ranked from highest to lowest model confidence",
        fontsize=10.2,
        color=TEXT_MUTED,
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


import math  # noqa: E402


def plot_confusion(model_data: dict, short_title_name: str, cmap_name: str, out_path: Path) -> None:
    cls = model_data["classification"]
    cats = cls["confusion_matrix_categories"]
    counts = np.array(cls["confusion_matrix_counts"], dtype=int)
    norm = np.array(cls["confusion_matrix_row_normalized"], dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(10.6, 8.5), dpi=160, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.21, right=0.89, top=0.84, bottom=0.16)

    ax.grid(False)
    im = ax.imshow(norm, cmap=cmap_name, vmin=0.0, vmax=100.0, aspect="auto")

    for i in range(len(cats)):
        for j in range(len(cats)):
            cnt = counts[i, j]
            pct = norm[i, j]
            if cnt == 0:
                ax.text(j, i, "0", ha="center", va="center", fontsize=8.5, color="#9CA3AF")
            else:
                txt_color = "#FFFFFF" if pct >= 48.0 else TEXT_DARK
                weight = "bold" if i == j else "normal"
                ax.text(
                    j,
                    i,
                    f"{pct:.0f}%\n({cnt})",
                    ha="center",
                    va="center",
                    fontsize=8.8,
                    fontweight=weight,
                    color=txt_color,
                )

    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels([SHORT_LABELS[c] for c in cats], fontsize=8.8)
    ax.set_yticks(np.arange(len(cats)))
    ax.set_yticklabels([ROW_LABELS[c] for c in cats], fontsize=9.8)
    ax.set_xlabel("Category picked by the model", fontsize=11.5, fontweight="bold", labelpad=10)
    ax.set_ylabel("Author's primary arXiv category (58 papers each)", fontsize=11.5, fontweight="bold", labelpad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
    cbar.set_label("Share of row's papers (%)", fontsize=10.2, labelpad=10)

    acc = cls["primary_accuracy"]["value"] * 100.0
    any_acc = cls["any_listed_category_accuracy"]["value"] * 100.0

    fig.suptitle(
        f"Where {short_title_name} agrees and disagrees with arXiv authors",
        fontsize=15.0,
        fontweight="bold",
        x=0.06,
        y=0.95,
        ha="left",
    )
    fig.text(
        0.06,
        0.898,
        f"464 unseen Nov 1-8, 2026 test papers — {acc:.1f}% exact primary match ({any_acc:.1f}% including cross-listed category)",
        fontsize=10.0,
        color=TEXT_MUTED,
        ha="left",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cover(metrics: dict, out_path: Path) -> None:
    """Generate exact 2000x800 (5:2) editorial cover image from real measured results."""
    drex = metrics["models"]["drex_max"]
    d1 = metrics["models"]["d1_2b"]

    fig = plt.figure(figsize=(20.0, 8.0), dpi=100, facecolor="#0F172A")
    # Exact 2000x800 canvas: left narrative & KPI cards, right auto-sort & 93%-claim plot
    ax_left = fig.add_axes([0.0, 0.0, 0.47, 1.0], facecolor="#0F172A")
    ax_left.axis("off")

    # Tag pill
    ax_left.text(
        0.075, 0.886, "OUT-OF-DISTRIBUTION CALIBRATION TEST  |  NOV 1-8, 2026 ARXIV",
        fontsize=11.0, fontweight="bold", color="#93C5FD", va="center", transform=ax_left.transAxes,
        bbox={"boxstyle": "round,pad=0.55,rounding_size=0.35", "facecolor": "#1E293B", "edgecolor": "#334155", "linewidth": 1.2},
    )

    # Headline
    ax_left.text(
        0.075, 0.755, "When Decision Models Say\nThey Are 93% Sure",
        fontsize=31.0, fontweight="bold", color="#F8FAFC", va="top", linespacing=1.18, transform=ax_left.transAxes,
    )
    ax_left.text(
        0.075, 0.535,
        "Testing Drex Max v1 (27B) and Liquid d1-2B (1.9B) on 464 unseen\n"
        "computer science preprints across 8 arXiv categories.",
        fontsize=14.5, color="#CBD5E1", va="top", linespacing=1.35, transform=ax_left.transAxes,
    )

    # Real KPI cards on bottom left
    drex_acc = drex["classification"]["primary_accuracy"]["value"] * 100.0
    d1_acc = d1["classification"]["primary_accuracy"]["value"] * 100.0

    # Native Drex claim (chance_corrected >= 0.93 out-of-the-box)
    drex_c93_cc = drex["claim_93"]["before_temp"]["chance_corrected"]
    drex_c93_val = (drex_c93_cc["accuracy_at_threshold"]["value"] or 0.0) * 100.0
    drex_c93_sh = drex_c93_cc["share_clearing"]["value"] * 100.0

    drex_cov95 = drex["coverage_at_95_precision"]["coverage"]["value"] * 100.0
    d1_cov95 = d1["coverage_at_95_precision"]["coverage"]["value"] * 100.0

    cards = [
        (0.068, "Overall Accuracy", f"{drex_acc:.1f}% vs {d1_acc:.1f}%", "Drex Max 27B vs d1-2B (n=464)", "#60A5FA"),
        (0.378, "Drex Max at 93%+ Sure", f"{drex_c93_val:.1f}% right", f"Covers {drex_c93_sh:.0f}% of test papers", "#34D399"),
        (0.688, "Auto-Sort at 95% Right", f"{drex_cov95:.0f}% vs {d1_cov95:.0f}%", "Share sorted without review", "#FBBF24"),
    ]
    for x0, title, big_num, sub, accent in cards:
        box = FancyBboxPatch(
            (x0, 0.09), 0.282, 0.30, boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#1E293B", edgecolor="#334155", linewidth=1.2, transform=ax_left.transAxes,
        )
        ax_left.add_patch(box)
        ax_left.text(x0 + 0.020, 0.34, title, fontsize=11.0, fontweight="bold", color="#94A3B8", va="top", transform=ax_left.transAxes)
        ax_left.text(x0 + 0.020, 0.245, big_num, fontsize=17.2, fontweight="bold", color=accent, va="center", transform=ax_left.transAxes)
        ax_left.text(x0 + 0.020, 0.135, sub, fontsize=9.8, color="#CBD5E1", va="bottom", transform=ax_left.transAxes)

    # Right chart panel: Coverage vs Accuracy curve on crisp light card
    ax_card = fig.add_axes([0.50, 0.09, 0.465, 0.82], facecolor="#0F172A")
    ax_card.axis("off")
    bg_box = FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor="#F8FAFC", edgecolor="#334155", linewidth=1.5, transform=ax_card.transAxes,
    )
    ax_card.add_patch(bg_box)

    ax = fig.add_axes([0.555, 0.20, 0.39, 0.59], facecolor="#FFFFFF")
    ax.grid(True, color="#E2E8F0", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#CBD5E1")

    min_k = 10
    for mdata, label, color in [
        (drex, "Drex Max v1 (27B)", "#1D4ED8"),
        (d1, "d1-2B (1.9B)", "#D97706"),
    ]:
        curve = mdata["coverage_at_95_precision"]["curve_primary"]
        cov = np.array(curve["coverage"]) * 100.0
        prec = np.array(curve["precision"]) * 100.0
        ks = np.array(curve["k"])
        mask = ks >= min_k
        ax.plot(cov[mask], prec[mask], color=color, linewidth=3.0, label=label, zorder=4)

        c95 = mdata["coverage_at_95_precision"]["coverage"]["value"] * 100.0
        if c95 > 0:
            ax.scatter([c95], [95.0], color=color, s=85, zorder=6, edgecolor="white", linewidth=1.6)

    ax.axhline(97.0, color="#DC2626", linestyle=":", linewidth=1.6, label="97% benchmark line (Nace report)")
    ax.axhline(95.0, color="#0F766E", linestyle="--", linewidth=1.6, label="95% auto-sort target")

    y_min = max(45.0, math.floor(min(drex_acc, d1_acc) / 5.0) * 5.0 - 8.0)
    ax.set_xlim(0, 102)
    ax.set_ylim(y_min, 102)
    ax.set_xlabel("Share of papers auto-sorted, most confident first (%)", fontsize=12.0, fontweight="bold", color="#0F172A")
    ax.set_ylabel("How often auto-sorted papers are right (%)", fontsize=12.0, fontweight="bold", color="#0F172A")
    ax.tick_params(colors="#334155", labelsize=10.5)
    ax.legend(loc="lower left", frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=10.2)
    ax.set_title(
        "Auto-sorting accuracy vs. share of papers handled automatically",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
        pad=12,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def generate_all_charts(metrics: dict, assets_dir: Path) -> None:
    setup_rc()
    assets_dir.mkdir(parents=True, exist_ok=True)

    drex = metrics["models"]["drex_max"]
    d1 = metrics["models"]["d1_2b"]

    plot_reliability(drex, "Drex Max", DREX_COLOR, DREX_AFTER, assets_dir / "reliability_drex.png")
    plot_reliability(d1, "d1-2B", D1_COLOR, D1_AFTER, assets_dir / "reliability_d1.png")
    plot_claim_93(metrics, assets_dir / "claim_93.png")
    plot_coverage_precision(metrics, assets_dir / "coverage_precision.png")
    plot_confusion(drex, "Drex Max v1 (27B)", "Blues", assets_dir / "confusion_drex.png")
    plot_confusion(d1, "d1-2B (1.9B)", "Oranges", assets_dir / "confusion_d1.png")
    plot_cover(metrics, assets_dir / "cover.png")

    print(f"Generated all 7 charts in {assets_dir.relative_to(ROOT)}/", flush=True)


def main() -> None:
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    generate_all_charts(metrics, ROOT / "assets")


if __name__ == "__main__":
    main()
