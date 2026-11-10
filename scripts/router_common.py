"""Shared prompt definitions and confidence helpers for the Part 2 LLM-router benchmark."""
from __future__ import annotations

import math
from collections.abc import Mapping

ROUTER_OPTIONS: list[str] = ["small", "frontier"]

ROUTER_CRITERIA: dict[str, str] = {
    "small": (
        "Compact fast LLM suffices: direct factual recall, single-step general Q&A, reading comprehension, "
        "or straightforward calculation that a lightweight 3B instruct model handles at frontier quality"
    ),
    "frontier": (
        "Escalate to frontier reasoning LLM: multi-step math or symbolic proof, competition or graduate STEM "
        "problem, intricate logic/coding deduction, or subtle edge cases where a 3B model frequently errs"
    ),
}

ROUTER_CRITERIA_REVERSED: dict[str, str] = dict(reversed(list(ROUTER_CRITERIA.items())))

ROUTER_QUESTION: dict[str, object] = {
    "type": "choice",
    "instructions": (
        "Should this user prompt be handled by a compact 3B LLM, or escalated to a frontier reasoning LLM "
        "to avoid losing answer quality?"
    ),
    "criteria": ROUTER_CRITERIA,
}

ROUTER_QUESTION_REVERSED: dict[str, object] = {
    "type": "choice",
    "instructions": (
        "Should this user prompt be handled by a compact 3B LLM, or escalated to a frontier reasoning LLM "
        "to avoid losing answer quality?"
    ),
    "criteria": ROUTER_CRITERIA_REVERSED,
}

NUM_ROUTER_OPTIONS: int = len(ROUTER_OPTIONS)  # 2

# Official SPROUT / CARROT per-1M-token USD pricing [input_per_1m, output_per_1m]
# Source: somerstep/CARROT (`carrot/constants.py`) + OpenAI API pricing for o3-mini ($1.10 / $4.40).
SPROUT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "wxai-llama-3-2-3b-instruct": (0.06, 0.06),
    "wxai-llama-3-1-8b-instruct": (0.20, 0.20),
    "openai-gpt-4o-mini": (0.15, 0.60),
    "openai-gpt-4o": (2.50, 10.00),
    "aws-claude-3-5-sonnet-v1": (3.00, 15.00),
    "openai-o3-mini": (1.10, 4.40),
}

PRIMARY_SMALL_MODEL = "wxai-llama-3-2-3b-instruct"
PRIMARY_FRONTIER_MODEL = "openai-o3-mini"


def format_router_state(prompt: str) -> str:
    """Format the raw user prompt as the decision-model state (zero answer/model leakage)."""
    return f"User prompt:\n{prompt.strip()}"


def compute_binary_confidences(probabilities: Mapping[str, float]) -> tuple[str, float, float, float]:
    """Return (choice, p_top, chance_corrected, p_frontier) from a {small, frontier} distribution."""
    total = sum(float(v) for v in probabilities.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError(f"Invalid probability distribution: {probabilities}")
    norm_probs = {opt: float(val) / total for opt, val in probabilities.items()}
    best_opt = max(norm_probs, key=norm_probs.__getitem__)
    p_top = norm_probs[best_opt]
    chance_corrected = (p_top - 0.5) / 0.5
    chance_corrected = max(0.0, min(1.0, chance_corrected))
    p_frontier = norm_probs["frontier"]
    return best_opt, p_top, chance_corrected, p_frontier
