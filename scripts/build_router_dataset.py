#!/usr/bin/env python3
"""Build a deterministic 600-query LLM-routing dataset (100 fit + 500 test) from CARROT-LLM-Routing/SPROUT-o3mini."""
from __future__ import annotations

import json
import math
import random
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from router_common import (  # noqa: E402
    PRIMARY_FRONTIER_MODEL,
    PRIMARY_SMALL_MODEL,
    SPROUT_MODEL_PRICING,
)

SPROUT_TEST_URL = (
    "https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT-o3mini/resolve/main/data/test-00000-of-00001.parquet"
)
MAX_PROMPT_CHARS = 4000
N_FIT = 100
N_TEST = 500
N_TOTAL = N_FIT + N_TEST
SEED = 42

TRACKED_MODELS = list(SPROUT_MODEL_PRICING.keys())


def map_family(dataset_name: str) -> str:
    if dataset_name.startswith("openhermes"):
        return "openhermes"
    if dataset_name.startswith("TIGER-Lab/MMLU-Pro"):
        return "MMLU-Pro"
    if dataset_name.startswith("lighteval/MATH"):
        return "MATH"
    if dataset_name.startswith("Idavidrein/gpqa"):
        return "GPQA"
    if dataset_name.startswith("TAUR-Lab/MuSR"):
        return "MuSR"
    if dataset_name.startswith("rungalileo/ragbench"):
        return "RAGBench"
    return dataset_name


def compute_cost_usd(model_name: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = SPROUT_MODEL_PRICING[model_name]
    return (in_tok * p_in + out_tok * p_out) / 1_000_000.0


def allocate_proportional(strata_counts: dict[tuple[str, str], int], target_n: int) -> dict[tuple[str, str], int]:
    """Largest-remainder proportional allocation matching exact strata proportions."""
    total = sum(strata_counts.values())
    raw = {k: (v / total) * target_n for k, v in strata_counts.items()}
    alloc = {k: int(math.floor(x)) for k, x in raw.items()}
    remainder = target_n - sum(alloc.values())
    ranked = sorted(raw.keys(), key=lambda k: (-(raw[k] - alloc[k]), k))
    for k in ranked[:remainder]:
        alloc[k] += 1
    assert sum(alloc.values()) == target_n
    return alloc


def main() -> None:
    scratch_parquet = (
        Path("/Users/davidflorez/.gemini/jetski/brain/169262f6-7ad6-4665-9897-4f6e1a6d17b9/scratch")
        / "sprout_o3mini_test.parquet"
    )
    if not scratch_parquet.exists():
        scratch_parquet.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {SPROUT_TEST_URL} -> {scratch_parquet}")
        req = urllib.request.Request(SPROUT_TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, scratch_parquet.open("wb") as fh:
            fh.write(resp.read())

    table = pq.read_table(scratch_parquet)
    n_all = table.num_rows
    keys = table["key"].to_pylist()
    datasets = table["dataset"].to_pylist()
    levels = table["dataset_level"].to_pylist()
    prompts = table["prompt"].to_pylist()
    model_cols = {m: table[m].to_pylist() for m in TRACKED_MODELS}

    eligible: list[dict] = []
    seen_keys: set[str] = set()
    for i in range(n_all):
        prompt = prompts[i]
        key = keys[i]
        if not prompt or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)

        fam = map_family(datasets[i])
        outcomes: dict[str, dict[str, float | int]] = {}
        for m in TRACKED_MODELS:
            entry = model_cols[m][i]
            in_t = int(entry["num_input_tokens"])
            out_t = int(entry["num_output_tokens"])
            sc = float(entry["score"])
            outcomes[m] = {
                "score": sc,
                "in_tokens": in_t,
                "out_tokens": out_t,
                "cost_usd": compute_cost_usd(m, in_t, out_t),
            }

        s_sm = outcomes[PRIMARY_SMALL_MODEL]["score"]
        s_fr = outcomes[PRIMARY_FRONTIER_MODEL]["score"]
        gold_route = "small" if s_sm >= s_fr else "frontier"

        eligible.append(
            {
                "id": f"sprout-{key[:12]}",
                "key": key,
                "dataset": datasets[i],
                "family": fam,
                "dataset_level": levels[i] or "",
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "gold_route": gold_route,
                "gold_needs_frontier": 0 if gold_route == "small" else 1,
                "outcomes": outcomes,
            }
        )

    rng = random.Random(SEED)
    by_stratum: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in eligible:
        by_stratum[(row["family"], row["gold_route"])].append(row)

    strata_counts = {k: len(v) for k, v in by_stratum.items()}
    total_alloc = allocate_proportional(strata_counts, N_TOTAL)

    selected: list[dict] = []
    selected_by_stratum: dict[tuple[str, str], list[dict]] = {}
    for k in sorted(by_stratum.keys()):
        pool = list(by_stratum[k])
        pool.sort(key=lambda r: r["key"])
        rng.shuffle(pool)
        take = pool[: total_alloc[k]]
        selected_by_stratum[k] = take
        selected.extend(take)

    # Split selected 600 stratified into 100 fit + 500 test
    selected_counts = {k: len(v) for k, v in selected_by_stratum.items()}
    fit_alloc = allocate_proportional(selected_counts, N_FIT)

    fit_rows: list[dict] = []
    test_rows: list[dict] = []
    for k in sorted(selected_by_stratum.keys()):
        items = list(selected_by_stratum[k])
        n_f = fit_alloc[k]
        for r in items[:n_f]:
            r_copy = dict(r)
            r_copy["split"] = "fit"
            fit_rows.append(r_copy)
        for r in items[n_f:]:
            r_copy = dict(r)
            r_copy["split"] = "test"
            test_rows.append(r_copy)

    # Sort deterministically by id within fit, then test
    fit_rows.sort(key=lambda r: r["id"])
    test_rows.sort(key=lambda r: r["id"])
    all_rows = fit_rows + test_rows
    assert len(fit_rows) == N_FIT and len(test_rows) == N_TEST and len(all_rows) == N_TOTAL

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = data_dir / "router_queries.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def summarize(rows: list[dict]) -> dict:
        n = len(rows)
        sm_scores = [r["outcomes"][PRIMARY_SMALL_MODEL]["score"] for r in rows]
        fr_scores = [r["outcomes"][PRIMARY_FRONTIER_MODEL]["score"] for r in rows]
        sm_costs = [r["outcomes"][PRIMARY_SMALL_MODEL]["cost_usd"] for r in rows]
        fr_costs = [r["outcomes"][PRIMARY_FRONTIER_MODEL]["cost_usd"] for r in rows]
        oracle_scores = [max(s, f) for s, f in zip(sm_scores, fr_scores)]
        oracle_costs = [
            r["outcomes"][PRIMARY_SMALL_MODEL]["cost_usd"]
            if r["gold_route"] == "small"
            else r["outcomes"][PRIMARY_FRONTIER_MODEL]["cost_usd"]
            for r in rows
        ]
        return {
            "n": n,
            "by_family": dict(sorted(Counter(r["family"] for r in rows).items())),
            "by_gold_route": dict(sorted(Counter(r["gold_route"] for r in rows).items())),
            "small_always_acc": sum(sm_scores) / n,
            "frontier_always_acc": sum(fr_scores) / n,
            "oracle_acc": sum(oracle_scores) / n,
            "small_always_cost_per_1k": 1000.0 * sum(sm_costs) / n,
            "frontier_always_cost_per_1k": 1000.0 * sum(fr_costs) / n,
            "oracle_cost_per_1k": 1000.0 * sum(oracle_costs) / n,
        }

    meta = {
        "source_dataset": "CARROT-LLM-Routing/SPROUT-o3mini",
        "source_split": "test",
        "source_rows_total": n_all,
        "eligible_rows_unique_under_4000_chars": len(eligible),
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "seed": SEED,
        "primary_small_model": PRIMARY_SMALL_MODEL,
        "primary_frontier_model": PRIMARY_FRONTIER_MODEL,
        "tracked_models_pricing_usd_per_1m": SPROUT_MODEL_PRICING,
        "fit": summarize(fit_rows),
        "test": summarize(test_rows),
        "total": summarize(all_rows),
    }
    meta_path = data_dir / "router_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
