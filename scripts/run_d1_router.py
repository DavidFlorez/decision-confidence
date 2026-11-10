#!/usr/bin/env python3
"""Part 2 (d1-2B Router) — Smoke test, 600-query canonical run, and reversed option-order run on MPS."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import PIL
import torch
import torchvision
import transformers
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from router_common import (  # noqa: E402
    ROUTER_OPTIONS,
    ROUTER_QUESTION,
    ROUTER_QUESTION_REVERSED,
    format_router_state,
)

MODEL_ID = "LiquidAI/d1-2B"
BATCH_SIZE = 16


def sync_device(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def extract_commit_hash(model: object) -> str:
    mod_file = inspect.getfile(model.__class__)
    matches = re.findall(r"\b([0-9a-f]{40})\b", mod_file)
    if matches:
        return matches[-1]
    hf_refs = Path.home() / ".cache" / "huggingface" / "hub" / "models--LiquidAI--d1-2B" / "refs" / "main"
    if hf_refs.exists():
        return hf_refs.read_text().strip()
    return f"unknown ({mod_file})"


def load_model() -> tuple[object, str, str]:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    print(f"[load] Attempting {MODEL_ID} on {device} ({dtype})...", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    return model, device, str(dtype).replace("torch.", "")


def evaluate_queries(
    model: object,
    device: str,
    queries: list[dict],
    question: dict,
    batch_size: int = BATCH_SIZE,
    label: str = "run",
) -> tuple[list[dict], float]:
    records: list[dict] = []
    expected_keys = set(ROUTER_OPTIONS)

    sync_device(device)
    t_total_start = time.perf_counter()

    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        requests = [(format_router_state(q["prompt"]), {"route": question}) for q in chunk]

        sync_device(device)
        t0 = time.perf_counter()
        outs = model.system_one_batch(requests)
        sync_device(device)
        batch_sec = time.perf_counter() - t0
        per_item_lat = batch_sec / len(chunk)

        if len(outs) != len(chunk):
            raise RuntimeError(f"Batch count mismatch: expected {len(chunk)}, got {len(outs)}")

        for q, out in zip(chunk, outs):
            ans = out["answers"]["route"]
            probs = {k: float(v) for k, v in ans["probabilities"].items()}
            choice = str(ans["choice"])
            conf = float(ans["confidence"])

            if set(probs.keys()) != expected_keys:
                raise RuntimeError(f"Probability keys {set(probs.keys())} != {expected_keys}")
            if any(not math.isfinite(v) or v < 0 for v in probs.values()):
                raise RuntimeError(f"Invalid probabilities for {q['id']}: {probs}")
            prob_sum = sum(probs.values())
            if abs(prob_sum - 1.0) > 1e-3:
                raise RuntimeError(f"Probabilities sum {prob_sum} != 1.0 for {q['id']}")
            max_prob = max(probs.values())
            if abs(conf - max_prob) > 1e-5:
                raise RuntimeError(f"Returned confidence {conf} != max_prob {max_prob} for {q['id']}")

            records.append(
                {
                    "id": q["id"],
                    "choice": choice,
                    "confidence": conf,
                    "probabilities": probs,
                    "latency": per_item_lat,
                }
            )

        done = start + len(chunk)
        print(
            f"[{label}] {done}/{len(queries)} queries | batch={batch_sec:.2f}s ({per_item_lat:.3f}s/query)",
            flush=True,
        )

    sync_device(device)
    total_wall_sec = time.perf_counter() - t_total_start
    return records, total_wall_sec


def write_jsonl(path: Path, records: list[dict], option_order: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            ordered_probs = {k: rec["probabilities"][k] for k in option_order}
            fh.write(
                json.dumps(
                    {
                        "id": rec["id"],
                        "choice": rec["choice"],
                        "confidence": rec["confidence"],
                        "probabilities": ordered_probs,
                        "latency": rec["latency"],
                    }
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiquidAI/d1-2B LLM-router evaluation.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()

    queries_path = ROOT / "data" / "router_queries.jsonl"
    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(queries)} router queries from {queries_path.relative_to(ROOT)}", flush=True)

    model, device, dtype_name = load_model()
    commit_hash = extract_commit_hash(model)
    cal_obj = getattr(model.engine, "calibration", None)

    env_info = {
        "model_id": MODEL_ID,
        "commit_hash": commit_hash,
        "served_default_temperature": getattr(cal_obj, "default", None),
        "device": device,
        "dtype": dtype_name,
        "batch_size": BATCH_SIZE,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "torchvision_version": torchvision.__version__,
        "pillow_version": PIL.__version__,
    }

    print("\n=== 20-Query Router Smoke Test ===", flush=True)
    smoke_20, smoke_20_sec = evaluate_queries(
        model, device, queries[:20], ROUTER_QUESTION, batch_size=BATCH_SIZE, label="smoke-20"
    )
    smoke_acc = sum(1 for q, r in zip(queries[:20], smoke_20) if r["choice"] == q["gold_route"]) / 20.0
    print(
        f"[smoke-20] wall={smoke_20_sec:.2f}s ({smoke_20_sec/20.0:.3f}s/query), sample_acc={smoke_acc:.3f}",
        flush=True,
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "router_d1_smoke.json").write_text(
        json.dumps(
            {
                "env": env_info,
                "smoke_20": {
                    "n": 20,
                    "wall_sec": smoke_20_sec,
                    "sec_per_query": smoke_20_sec / 20.0,
                    "projected_600_minutes": (smoke_20_sec / 20.0 * len(queries)) / 60.0,
                    "sample_accuracy_20": smoke_acc,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.mode == "smoke":
        return

    print(f"\n=== Full Canonical Run: {len(queries)} queries ===", flush=True)
    main_records, main_wall_sec = evaluate_queries(
        model, device, queries, ROUTER_QUESTION, batch_size=BATCH_SIZE, label="d1-router"
    )
    write_jsonl(results_dir / "router_d1_2b.jsonl", main_records, ROUTER_OPTIONS)

    print(f"\n=== Reversed Option-Order Run: {len(queries)} queries ===", flush=True)
    rev_records, rev_wall_sec = evaluate_queries(
        model, device, queries, ROUTER_QUESTION_REVERSED, batch_size=BATCH_SIZE, label="d1-router-rev"
    )
    rev_opts = list(ROUTER_QUESTION_REVERSED["criteria"].keys())
    write_jsonl(results_dir / "router_d1_2b_reversed.jsonl", rev_records, rev_opts)

    meta = {
        "env": env_info,
        "n_queries": len(queries),
        "main_wall_sec": main_wall_sec,
        "main_sec_per_query": main_wall_sec / len(queries),
        "reversed_wall_sec": rev_wall_sec,
        "reversed_sec_per_query": rev_wall_sec / len(queries),
    }
    (results_dir / "router_d1_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("\n[done] Router d1-2B inference complete.", flush=True)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
