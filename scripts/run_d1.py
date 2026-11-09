#!/usr/bin/env python3
"""Step 2 & Step 4 (d1-2B) — Smoke test, full 560-paper run, and reversed option-order run on MPS."""
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

from common import CATEGORIES, QUESTION, QUESTION_REVERSED, format_state  # noqa: E402

MODEL_ID = "LiquidAI/d1-2B"
BATCH_SIZE = 16


def sync_device(device: str) -> None:
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def extract_commit_hash(model: object) -> str:
    """Extract the exact Hugging Face snapshot commit hash from the loaded remote-code module path."""
    mod_file = inspect.getfile(model.__class__)
    # Path typically contains .../snapshots/<40-hex-sha>/... or .../transformers_modules/.../<40-hex-sha>/...
    matches = re.findall(r"\b([0-9a-f]{40})\b", mod_file)
    if matches:
        return matches[-1]
    # Also check HF cache refs/main if needed
    hf_refs = Path.home() / ".cache" / "huggingface" / "hub" / "models--LiquidAI--d1-2B" / "refs" / "main"
    if hf_refs.exists():
        return hf_refs.read_text().strip()
    return f"unknown ({mod_file})"


def run_ticket_smoke_test(model: object, device: str) -> dict:
    """Reproduce the LiquidAI/d1-2B model card quick-start ticket example."""
    questions = {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges, refunds, invoices",
                "shipping": "Delivery and parcels",
                "technical": "App or site faults",
            },
        },
        "topics": {
            "type": "choice",
            "multiple": True,
            "instructions": "Which topics does the message raise?",
            "criteria": {
                "billing": "Payments and charges",
                "delivery": "Shipping and parcels",
                "login": "Signing in to the account",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["low", "medium", "high"],
        },
    }
    ticket = "I was charged twice and my parcel never arrived. Please refund one charge."
    sync_device(device)
    t0 = time.perf_counter()
    out = model.system_one(ticket, questions)
    sync_device(device)
    elapsed = time.perf_counter() - t0

    answers = out["answers"]
    team_ans = answers["team"]
    probs = team_ans["probabilities"]
    prob_sum = sum(probs.values())
    if not all(math.isfinite(v) and v >= 0 for v in probs.values()):
        raise RuntimeError(f"Non-finite or negative probabilities in ticket test: {probs}")
    if team_ans["choice"] != "billing":
        raise RuntimeError(f"Ticket smoke test expected choice='billing', got {team_ans!r}")
    if abs(prob_sum - 1.0) > 1e-3:
        raise RuntimeError(f"Ticket probabilities sum {prob_sum} != 1.0: {probs}")

    return {
        "elapsed_sec": elapsed,
        "answers": answers,
        "usage": out.get("usage"),
    }


def load_model_with_fallback() -> tuple[object, str, str, dict, list[str]]:
    """Try mps+bfloat16 -> mps+float32 -> cpu+float32, verifying via ticket smoke test."""
    candidates: list[tuple[str, torch.dtype, str]] = []
    if torch.backends.mps.is_available():
        candidates.append(("mps", torch.bfloat16, "bfloat16"))
        candidates.append(("mps", torch.float32, "float32"))
    candidates.append(("cpu", torch.float32, "float32"))

    fallback_log: list[str] = []
    for device, dtype, dtype_name in candidates:
        try:
            print(f"[load] Trying {MODEL_ID} on device={device}, dtype={dtype_name}...", flush=True)
            t0 = time.perf_counter()
            model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype).to(device)
            model.eval()
            load_sec = time.perf_counter() - t0
            print(f"[load] Loaded in {load_sec:.2f}s. Verifying API & ticket smoke test...", flush=True)

            # Verify remote API method names against SystemOneModel -> engine (SystemOneApi.system_one_batch)
            model_mro = [cls.__name__ for cls in model.__class__.__mro__]
            engine_mro = [cls.__name__ for cls in model.engine.__class__.__mro__]
            if (
                "SystemOneModel" not in model_mro
                or "SystemOneApi" not in engine_mro
                or not hasattr(model, "system_one_batch")
                or not hasattr(model.engine, "system_one_batch")
            ):
                raise RuntimeError(
                    f"Expected SystemOneModel -> SystemOneApi.system_one_batch; got model={model_mro}, engine={engine_mro}"
                )

            ticket_res = run_ticket_smoke_test(model, device)
            ticket_res["load_sec"] = load_sec
            fallback_log.append(f"SUCCESS: device={device}, dtype={dtype_name}")
            return model, device, dtype_name, ticket_res, fallback_log
        except Exception as exc:
            msg = f"FAILED: device={device}, dtype={dtype_name} -> {type(exc).__name__}: {exc}"
            print(f"[warn] {msg}", flush=True)
            fallback_log.append(msg)

    raise RuntimeError(f"All device/dtype candidates failed: {fallback_log}")


def evaluate_papers(
    model: object,
    device: str,
    papers: list[dict],
    question: dict,
    batch_size: int = BATCH_SIZE,
    label: str = "eval",
) -> tuple[list[dict], float]:
    """Run system_one_batch in chunks of batch_size and return (records, total_wall_sec)."""
    records: list[dict] = []
    expected_keys = set(question["criteria"].keys())

    sync_device(device)
    t_total_start = time.perf_counter()

    for start in range(0, len(papers), batch_size):
        chunk = papers[start:start + batch_size]
        requests = [(format_state(p["title"], p["abstract"]), {"category": question}) for p in chunk]

        sync_device(device)
        t0 = time.perf_counter()
        outs = model.system_one_batch(requests)
        sync_device(device)
        batch_sec = time.perf_counter() - t0
        per_paper_lat = batch_sec / len(chunk)

        if len(outs) != len(chunk):
            raise RuntimeError(f"Batch output count mismatch: expected {len(chunk)}, got {len(outs)}")

        for paper, out in zip(chunk, outs):
            ans = out["answers"]["category"]
            probs = {k: float(v) for k, v in ans["probabilities"].items()}
            choice = str(ans["choice"])
            conf = float(ans["confidence"])

            if set(probs.keys()) != expected_keys:
                raise RuntimeError(f"Probability keys {set(probs.keys())} != {expected_keys}")
            if any(not math.isfinite(v) or v < 0 for v in probs.values()):
                raise RuntimeError(f"Invalid probabilities for {paper['id']}: {probs}")
            prob_sum = sum(probs.values())
            if abs(prob_sum - 1.0) > 1e-3:
                raise RuntimeError(f"Probabilities sum to {prob_sum} (expected ~1.0) for {paper['id']}")
            max_prob = max(probs.values())
            if abs(conf - max_prob) > 1e-5:
                raise RuntimeError(
                    f"Returned confidence {conf} != max_prob {max_prob} for {paper['id']}"
                )

            records.append(
                {
                    "id": paper["id"],
                    "choice": choice,
                    "confidence": conf,
                    "probabilities": probs,
                    "latency": per_paper_lat,
                }
            )

        done = start + len(chunk)
        print(
            f"[{label}] {done}/{len(papers)} papers | batch={batch_sec:.2f}s ({per_paper_lat:.3f}s/paper)",
            flush=True,
        )

    sync_device(device)
    total_wall_sec = time.perf_counter() - t_total_start
    return records, total_wall_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LiquidAI/d1-2B smoke test and full evaluation.")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()

    papers_path = ROOT / "data" / "papers.jsonl"
    papers = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(papers)} papers from {papers_path.relative_to(ROOT)}", flush=True)

    model, device, dtype_name, ticket_res, fallback_log = load_model_with_fallback()
    commit_hash = extract_commit_hash(model)
    mro_names = [f"{c.__module__}.{c.__qualname__}" for c in model.__class__.__mro__[:5]]
    engine_mro_names = [f"{c.__module__}.{c.__qualname__}" for c in model.engine.__class__.__mro__]
    batch_sig = str(inspect.signature(model.engine.system_one_batch))
    cal_obj = getattr(model.engine, "calibration", None)
    served_temp = getattr(cal_obj, "default", None)
    served_bucket_temps = getattr(cal_obj, "temperatures", None)

    env_info = {
        "model_id": MODEL_ID,
        "commit_hash": commit_hash,
        "served_default_temperature": served_temp,
        "served_bucket_temperatures": served_bucket_temps,
        "device": device,
        "dtype": dtype_name,
        "batch_size": BATCH_SIZE,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "torchvision_version": torchvision.__version__,
        "pillow_version": PIL.__version__,
        "mro_head": mro_names,
        "engine_mro": engine_mro_names,
        "system_one_batch_signature": batch_sig,
        "fallback_log": fallback_log,
    }
    print("\n=== Environment & Model Verification ===", flush=True)
    print(json.dumps(env_info, indent=2), flush=True)
    print("\n=== Ticket Quick-Start Smoke Test ===", flush=True)
    print(json.dumps(ticket_res, indent=2), flush=True)

    # Smoke test on 20 dataset papers
    print("\n=== 20-Paper Dataset Smoke Test ===", flush=True)
    smoke_20, smoke_20_sec = evaluate_papers(
        model, device, papers[:20], QUESTION, batch_size=BATCH_SIZE, label="smoke-20"
    )
    sec_per_paper = smoke_20_sec / 20.0
    est_full_min = (sec_per_paper * len(papers)) / 60.0
    max_sum_err = max(abs(sum(r["probabilities"].values()) - 1.0) for r in smoke_20)
    max_conf_diff = max(abs(r["confidence"] - max(r["probabilities"].values())) for r in smoke_20)
    smoke_correct = sum(1 for p, r in zip(papers[:20], smoke_20) if r["choice"] == p["primary"])

    smoke_summary = {
        "env": env_info,
        "ticket_test": ticket_res,
        "smoke_20": {
            "n": 20,
            "wall_sec": smoke_20_sec,
            "sec_per_paper": sec_per_paper,
            "projected_560_minutes": est_full_min,
            "max_prob_sum_abs_err": max_sum_err,
            "max_conf_vs_max_prob_diff": max_conf_diff,
            "sample_accuracy_20": smoke_correct / 20.0,
        },
    }
    print(json.dumps(smoke_summary["smoke_20"], indent=2), flush=True)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "d1_smoke_test.json").write_text(
        json.dumps(smoke_summary, indent=2) + "\n", encoding="utf-8"
    )

    if args.mode == "smoke":
        print("Smoke test passed cleanly (--mode smoke).", flush=True)
        return

    if est_full_min > 30.0:
        print(
            f"[note] Projected 560-paper run is {est_full_min:.1f} min (>30 min); proceeding and logging runtime.",
            flush=True,
        )

    # Full 560-paper canonical-order run
    print(f"\n=== Full Run: {len(papers)} papers (canonical option order) ===", flush=True)
    full_records, full_wall_sec = evaluate_papers(
        model, device, papers, QUESTION, batch_size=BATCH_SIZE, label="d1-full"
    )
    out_main = results_dir / "d1_2b.jsonl"
    with out_main.open("w", encoding="utf-8") as fh:
        for rec in full_records:
            # Ensure canonical category key order in saved probabilities dict
            ordered_probs = {c: rec["probabilities"][c] for c in CATEGORIES}
            row_out = {
                "id": rec["id"],
                "choice": rec["choice"],
                "confidence": rec["confidence"],
                "probabilities": ordered_probs,
                "latency": rec["latency"],
            }
            fh.write(json.dumps(row_out) + "\n")

    # Step 4: Option-order check (reversed options on all 560 papers; test split analyzed in analyze.py)
    print(f"\n=== Reversed Option-Order Run: {len(papers)} papers ===", flush=True)
    rev_records, rev_wall_sec = evaluate_papers(
        model, device, papers, QUESTION_REVERSED, batch_size=BATCH_SIZE, label="d1-reversed"
    )
    rev_cats = list(QUESTION_REVERSED["criteria"].keys())
    out_rev = results_dir / "d1_2b_reversed.jsonl"
    with out_rev.open("w", encoding="utf-8") as fh:
        for rec in rev_records:
            ordered_probs = {c: rec["probabilities"][c] for c in rev_cats}
            row_out = {
                "id": rec["id"],
                "choice": rec["choice"],
                "confidence": rec["confidence"],
                "probabilities": ordered_probs,
                "latency": rec["latency"],
            }
            fh.write(json.dumps(row_out) + "\n")

    run_meta = {
        **smoke_summary,
        "full_run": {
            "n_papers": len(full_records),
            "wall_sec": full_wall_sec,
            "papers_per_sec": len(full_records) / full_wall_sec,
            "mean_latency_sec": sum(r["latency"] for r in full_records) / len(full_records),
        },
        "reversed_run": {
            "n_papers": len(rev_records),
            "wall_sec": rev_wall_sec,
            "papers_per_sec": len(rev_records) / rev_wall_sec,
            "mean_latency_sec": sum(r["latency"] for r in rev_records) / len(rev_records),
        },
    }
    (results_dir / "d1_run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nSaved {out_main.relative_to(ROOT)} ({full_wall_sec:.1f}s, "
        f"{len(full_records) / full_wall_sec:.2f} papers/s) and {out_rev.relative_to(ROOT)}!",
        flush=True,
    )


if __name__ == "__main__":
    main()
