#!/usr/bin/env python3
"""Part 2 (Drex Max v1 Router) — Smoke test, 600-query canonical run, and reversed option-order run on CUDA bf16."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from router_common import (  # noqa: E402
    ROUTER_OPTIONS,
    ROUTER_QUESTION,
    ROUTER_QUESTION_REVERSED,
    format_router_state,
)


def collect_gpu_and_env_info(weights_dir: Path) -> dict:
    import torch
    import transformers

    try:
        import fla
        fla_version = getattr(fla, "__version__", "installed (no __version__)")
    except Exception as exc:
        fla_version = f"unavailable ({exc})"

    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True, timeout=15).strip()
    except Exception as exc:
        smi = f"nvidia-smi failed: {exc}"

    dec_cfg_path = weights_dir / "decision_config.json"
    dec_cfg = json.loads(dec_cfg_path.read_text(encoding="utf-8")) if dec_cfg_path.exists() else {}
    hf_info_path = weights_dir / "hf_download_info.json"
    hf_info = json.loads(hf_info_path.read_text(encoding="utf-8")) if hf_info_path.exists() else {}

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if torch.cuda.is_available()
        else None
    )

    return {
        "model_id": "nace-ai/drex-max-v1",
        "repo_commit_sha": hf_info.get("commit_sha"),
        "download_wall_sec": hf_info.get("download_wall_sec"),
        "checkpoint_base_model": dec_cfg.get("base_model"),
        "checkpoint_revision": dec_cfg.get("revision"),
        "served_default_temperature": dec_cfg.get("temperature"),
        "attention_mode": dec_cfg.get("attention_mode"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "fla_version": fla_version,
        "gpu_name": gpu_name,
        "gpu_vram_gib": vram_gb,
        "nvidia_smi": smi,
    }


def evaluate_queries(
    model: object,
    answer_fn: object,
    queries: list[dict],
    question: dict,
    batch_size: int,
    label: str,
) -> tuple[list[dict], float]:
    import torch

    records: list[dict] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_total0 = time.perf_counter()

    for start in range(0, len(queries), batch_size):
        chunk = queries[start : start + batch_size]
        rows = [{"state": format_router_state(q["prompt"]), "question": question} for q in chunk]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs_batch = model.predict(rows, batch_size=batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_sec = time.perf_counter() - t0
        per_item_lat = batch_sec / len(chunk)

        for q, p_vec in zip(chunk, probs_batch):
            raw_sum = sum(float(x) for x in p_vec)
            if not all(math.isfinite(x) and x >= 0 for x in p_vec) or abs(raw_sum - 1.0) > 1e-3:
                raise RuntimeError(f"Invalid probability vector (sum={raw_sum}) for {q['id']}: {p_vec}")
            formatted = answer_fn(question, p_vec)
            records.append(
                {
                    "id": q["id"],
                    "choice": str(formatted["choice"]),
                    "confidence": float(formatted["confidence"]),
                    "probabilities": {k: float(v) for k, v in formatted["probabilities"].items()},
                    "latency": per_item_lat,
                }
            )

        done = start + len(chunk)
        print(
            f"[{label}] {done}/{len(queries)} queries | batch={batch_sec:.2f}s ({per_item_lat:.3f}s/query)",
            flush=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_wall_sec = time.perf_counter() - t_total0
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
    parser = argparse.ArgumentParser(description="Run Drex Max v1 on the 600-query SPROUT router benchmark.")
    parser.add_argument("--weights", default=str(ROOT / "weights" / "drex-max-v1"))
    parser.add_argument("--queries", default=str(ROOT / "data" / "router_queries.jsonl"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()

    weights_dir = Path(args.weights).resolve()
    queries_path = Path(args.queries).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    env_info = collect_gpu_and_env_info(weights_dir)
    env_info["batch_size"] = args.batch_size
    print("=== Environment & GPU Info ===", flush=True)
    print(json.dumps({k: v for k, v in env_info.items() if k != "nvidia_smi"}, indent=2), flush=True)

    sys.path.insert(0, str(weights_dir / "code"))
    from drex_max.model import DecisionModel, answer  # noqa: E402

    print(f"[load] Loading DecisionModel({weights_dir}) in bf16...", flush=True)
    t_load0 = time.perf_counter()
    model = DecisionModel(str(weights_dir))
    load_sec = time.perf_counter() - t_load0
    print(f"[load] Loaded in {load_sec:.2f}s (device={model.device_name}, temp={model.temperature:.6f})", flush=True)

    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(queries)} router queries.", flush=True)

    print("[warmup] Running 1 warmup query to initialize kernels...", flush=True)
    _ = evaluate_queries(model, answer, queries[:1], ROUTER_QUESTION, batch_size=1, label="warmup")

    print("\n=== 50-Query Router Smoke Test ===", flush=True)
    smoke_50, smoke_50_sec = evaluate_queries(
        model, answer, queries[:50], ROUTER_QUESTION, batch_size=args.batch_size, label="smoke-50"
    )
    sec_per_req = smoke_50_sec / 50.0
    smoke_acc = sum(1 for q, r in zip(queries[:50], smoke_50) if r["choice"] == q["gold_route"]) / 50.0
    smoke_summary = {
        "env": env_info,
        "load_sec": load_sec,
        "smoke_50": {
            "n": 50,
            "wall_sec": smoke_50_sec,
            "sec_per_query": sec_per_req,
            "sample_accuracy_50": smoke_acc,
        },
    }
    print(json.dumps(smoke_summary["smoke_50"], indent=2), flush=True)
    (results_dir / "router_drex_smoke.json").write_text(json.dumps(smoke_summary, indent=2) + "\n", encoding="utf-8")

    if args.mode == "smoke":
        return

    print(f"\n=== Full Canonical Run: {len(queries)} queries ===", flush=True)
    main_records, main_wall_sec = evaluate_queries(
        model, answer, queries, ROUTER_QUESTION, batch_size=args.batch_size, label="drex-router"
    )
    write_jsonl(results_dir / "router_drex_max.jsonl", main_records, ROUTER_OPTIONS)

    print(f"\n=== Reversed Option-Order Run: {len(queries)} queries ===", flush=True)
    rev_records, rev_wall_sec = evaluate_queries(
        model, answer, queries, ROUTER_QUESTION_REVERSED, batch_size=args.batch_size, label="drex-router-rev"
    )
    rev_opts = list(ROUTER_QUESTION_REVERSED["criteria"].keys())
    write_jsonl(results_dir / "router_drex_max_reversed.jsonl", rev_records, rev_opts)

    run_meta = {
        "env": env_info,
        "load_sec": load_sec,
        "smoke_50": smoke_summary["smoke_50"],
        "n_queries": len(queries),
        "main_wall_sec": main_wall_sec,
        "main_sec_per_query": main_wall_sec / len(queries),
        "reversed_wall_sec": rev_wall_sec,
        "reversed_sec_per_query": rev_wall_sec / len(queries),
    }
    (results_dir / "router_drex_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    print("\n[done] Router Drex Max inference complete.", flush=True)
    print(json.dumps({k: v for k, v in run_meta.items() if k != "env"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
