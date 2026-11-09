#!/usr/bin/env python3
"""Step 3 & Step 4 (Drex Max v1) — Smoke tests, full 560-paper run, and reversed option-order run on CUDA bf16."""
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

from common import CATEGORIES, QUESTION, QUESTION_REVERSED, format_state  # noqa: E402

EXAMPLE_REQUEST = {
    "state": {
        "ticket": "I was charged twice for the same order. Please refund the extra payment."
    },
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, charges, and refunds",
                "technical": "Bugs and outages",
                "other": "Anything else",
            },
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer explicitly ask for a refund?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["Routine", "Soon", "Urgent"],
        },
    },
}


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


def run_cli_smoke_test(weights_dir: Path, request_path: Path) -> dict:
    """Run weights/drex-max-v1/inference.py on Nace's examples/request.json and check H100 bf16 reference."""
    request_path.write_text(json.dumps(EXAMPLE_REQUEST, indent=2) + "\n", encoding="utf-8")
    inference_py = weights_dir / "inference.py"
    if not inference_py.exists():
        raise FileNotFoundError(f"Missing {inference_py}")

    cmd = [sys.executable, str(inference_py), "--model", str(weights_dir), "--request", str(request_path)]
    print(f"[smoke-cli] Running: {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"inference.py failed (exit {proc.returncode}):\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")

    # Parse JSON object from stdout
    stdout = proc.stdout.strip()
    json_start = stdout.find("{")
    if json_start < 0:
        raise RuntimeError(f"Could not find JSON in inference.py stdout:\n{stdout}")
    parsed = json.loads(stdout[json_start:])
    ans = parsed["answers"]

    team_choice = ans["team"]["choice"]
    p_billing = float(ans["team"]["probabilities"]["billing"])
    team_conf = float(ans["team"]["confidence"])
    p_refund = float(ans["refund"]["noul"])

    checks = {
        "team_choice_is_billing": team_choice == "billing",
        "billing_prob_diff": abs(p_billing - 0.9988),
        "confidence_diff": abs(team_conf - 0.9982),
        "refund_diff": abs(p_refund - 0.9974),
    }

    if team_choice != "billing":
        raise RuntimeError(f"STOP: Expected team='billing', got {team_choice!r}. Full output: {parsed}")
    for key in ("billing_prob_diff", "confidence_diff", "refund_diff"):
        if checks[key] > 0.03:
            raise RuntimeError(
                f"STOP: Smoke test {key}={checks[key]:.4f} > 0.03 vs H100 bf16 reference! "
                f"Likely Blackwell + flash-linear-attention kernel or transformers version issue. Output: {parsed}"
            )

    return {
        "elapsed_sec": elapsed,
        "output": parsed,
        "reference_diffs": checks,
    }


def evaluate_papers(
    model: object,
    answer_fn: object,
    papers: list[dict],
    question: dict,
    batch_size: int = 8,
    label: str = "drex",
) -> tuple[list[dict], float]:
    import torch

    records: list[dict] = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_total_start = time.perf_counter()

    for start in range(0, len(papers), batch_size):
        chunk = papers[start:start + batch_size]
        rows = [{"state": format_state(p["title"], p["abstract"]), "question": question} for p in chunk]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs_batch = model.predict(rows, batch_size=batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_sec = time.perf_counter() - t0
        per_paper_lat = batch_sec / len(chunk)

        for paper, p_vec in zip(chunk, probs_batch):
            raw_sum = sum(float(x) for x in p_vec)
            if not all(math.isfinite(x) and x >= 0 for x in p_vec) or abs(raw_sum - 1.0) > 1e-3:
                raise RuntimeError(f"Invalid probability vector (sum={raw_sum}) for {paper['id']}: {p_vec}")
            formatted = answer_fn(question, p_vec)
            records.append(
                {
                    "id": paper["id"],
                    "choice": str(formatted["choice"]),
                    "confidence": float(formatted["confidence"]),
                    "probabilities": {k: float(v) for k, v in formatted["probabilities"].items()},
                    "latency": per_paper_lat,
                }
            )

        done = start + len(chunk)
        print(
            f"[{label}] {done}/{len(papers)} papers | batch={batch_sec:.2f}s ({per_paper_lat:.3f}s/paper)",
            flush=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_wall_sec = time.perf_counter() - t_total_start
    return records, total_wall_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Drex Max v1 smoke tests, full 560-paper run, and reorder check.")
    parser.add_argument("--weights", default=str(ROOT / "weights" / "drex-max-v1"))
    parser.add_argument("--papers", default=str(ROOT / "data" / "papers.jsonl"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--skip-cli-smoke", action="store_true", help="Skip subprocess CLI smoke check if already verified")
    args = parser.parse_args()

    weights_dir = Path(args.weights).resolve()
    papers_path = Path(args.papers).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    env_info = collect_gpu_and_env_info(weights_dir)
    env_info["batch_size"] = args.batch_size
    print("=== Environment & GPU Info ===", flush=True)
    print(json.dumps({k: v for k, v in env_info.items() if k != "nvidia_smi"}, indent=2), flush=True)
    print(env_info["nvidia_smi"], flush=True)

    cli_smoke = None
    if not args.skip_cli_smoke:
        req_path = results_dir / "request_smoke.json"
        cli_smoke = run_cli_smoke_test(weights_dir, req_path)
        print("\n=== CLI Smoke Test Passed ===", flush=True)
        print(json.dumps(cli_smoke, indent=2), flush=True)

    # Verify bundled loader symbols in weights/drex-max-v1/code/drex_max/model.py
    sys.path.insert(0, str(weights_dir / "code"))
    import drex_max.model as drex_mod
    from drex_max.model import DecisionModel, answer

    for sym in ("DecisionModel", "answer"):
        if not hasattr(drex_mod, sym):
            raise RuntimeError(f"Missing symbol {sym} in {drex_mod.__file__}")
    if not hasattr(DecisionModel, "predict"):
        raise RuntimeError("DecisionModel has no .predict method")

    print(
        f"\n[verify] Verified DecisionModel, DecisionModel.predict{inspect.signature(DecisionModel.predict)}, "
        f"and answer{inspect.signature(answer)} in {drex_mod.__file__}",
        flush=True,
    )

    print(f"[load] Loading DecisionModel({weights_dir}) in bf16...", flush=True)
    t_load0 = time.perf_counter()
    model = DecisionModel(str(weights_dir))
    load_sec = time.perf_counter() - t_load0
    print(f"[load] Loaded in {load_sec:.2f}s (device={model.device_name}, temp={model.temperature:.6f})", flush=True)

    papers = [json.loads(line) for line in papers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(papers)} dataset papers.", flush=True)

    # Warmup 1 batch so CUDA / Triton / FLA kernels compile before timing 50-paper smoke check
    print("[warmup] Running 1 warmup paper to initialize kernels...", flush=True)
    _ = evaluate_papers(model, answer, papers[:1], QUESTION, batch_size=1, label="warmup")

    # 50-paper dataset smoke test
    print("\n=== 50-Paper Dataset Smoke Test ===", flush=True)
    smoke_50, smoke_50_sec = evaluate_papers(
        model, answer, papers[:50], QUESTION, batch_size=args.batch_size, label="smoke-50"
    )
    sec_per_req = smoke_50_sec / 50.0
    max_sum_err = max(abs(sum(r["probabilities"].values()) - 1.0) for r in smoke_50)
    smoke_50_acc = sum(1 for p, r in zip(papers[:50], smoke_50) if r["choice"] == p["primary"]) / 50.0

    smoke_50_summary = {
        "n": 50,
        "wall_sec": smoke_50_sec,
        "sec_per_paper": sec_per_req,
        "max_prob_sum_abs_err": max_sum_err,
        "sample_accuracy_50": smoke_50_acc,
    }
    print(json.dumps(smoke_50_summary, indent=2), flush=True)

    if sec_per_req > 1.0:
        raise RuntimeError(
            f"STOP: 50-paper smoke test took {sec_per_req:.3f}s/paper (> 1.0s limit)! "
            f"Investigate batch_size and FLA/Blackwell kernels before full run."
        )

    smoke_meta = {
        "env": env_info,
        "load_sec": load_sec,
        "cli_smoke": cli_smoke,
        "smoke_50": smoke_50_summary,
    }
    (results_dir / "drex_smoke_test.json").write_text(json.dumps(smoke_meta, indent=2) + "\n", encoding="utf-8")

    if args.mode == "smoke":
        print("Drex Max smoke tests passed cleanly (--mode smoke).", flush=True)
        return

    # Full 560-paper canonical-order run
    print(f"\n=== Full Run: {len(papers)} papers (canonical option order) ===", flush=True)
    full_records, full_wall_sec = evaluate_papers(
        model, answer, papers, QUESTION, batch_size=args.batch_size, label="drex-full"
    )
    out_main = results_dir / "drex_max.jsonl"
    with out_main.open("w", encoding="utf-8") as fh:
        for rec in full_records:
            ordered_probs = {c: rec["probabilities"][c] for c in CATEGORIES}
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

    # Step 4: Reversed option-order run
    print(f"\n=== Reversed Option-Order Run: {len(papers)} papers ===", flush=True)
    rev_records, rev_wall_sec = evaluate_papers(
        model, answer, papers, QUESTION_REVERSED, batch_size=args.batch_size, label="drex-reversed"
    )
    rev_cats = list(QUESTION_REVERSED["criteria"].keys())
    out_rev = results_dir / "drex_max_reversed.jsonl"
    with out_rev.open("w", encoding="utf-8") as fh:
        for rec in rev_records:
            ordered_probs = {c: rec["probabilities"][c] for c in rev_cats}
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

    run_meta = {
        **smoke_meta,
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
    (results_dir / "drex_run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"\nSaved {out_main} ({full_wall_sec:.1f}s, {len(full_records) / full_wall_sec:.2f} papers/s) "
        f"and {out_rev}!",
        flush=True,
    )


if __name__ == "__main__":
    main()
