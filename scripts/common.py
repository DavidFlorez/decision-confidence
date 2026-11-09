"""Shared constants and formatting utilities for the decision-confidence study."""
from __future__ import annotations

import math
from collections.abc import Mapping

# Primary 8-category evaluation set (Set B), in exact canonical order.
CATEGORIES: list[str] = [
    "cs.CL",
    "cs.CV",
    "cs.LG",
    "cs.RO",
    "cs.CR",
    "cs.SE",
    "cs.DC",
    "cs.HC",
]

# Alternative 8-category set (Set A) checked for cross-list overlap comparison.
ALT_CATEGORIES_SET_A: list[str] = [
    "cs.CL",
    "cs.CV",
    "cs.LG",
    "cs.AI",
    "cs.RO",
    "cs.CR",
    "cs.SE",
    "cs.IR",
]

# Union of all categories queried from arXiv API so both Set B and Set A counts are exact.
ALL_QUERY_CATEGORIES: list[str] = [
    "cs.CL",
    "cs.CV",
    "cs.LG",
    "cs.RO",
    "cs.CR",
    "cs.SE",
    "cs.DC",
    "cs.HC",
    "cs.AI",
    "cs.IR",
]

# Exact one-line descriptions in canonical order.
CRITERIA: dict[str, str] = {
    "cs.CL": "Computation and Language: natural language processing, language models, speech and text",
    "cs.CV": "Computer Vision: images, video, 3D vision and visual generation",
    "cs.LG": "Machine Learning: learning algorithms, optimization and theory not tied to one application",
    "cs.RO": "Robotics: robot control, manipulation, locomotion, navigation and embodied agents",
    "cs.CR": "Cryptography and Security: attacks, defenses, privacy and cryptography",
    "cs.SE": "Software Engineering: building, testing and maintaining software, and tools for code",
    "cs.DC": "Distributed, Parallel and Cluster Computing: distributed systems, HPC, scheduling, serving at scale",
    "cs.HC": "Human-Computer Interaction: user studies, interfaces and how people use technology",
}

CRITERIA_REVERSED: dict[str, str] = dict(reversed(list(CRITERIA.items())))

QUESTION: dict[str, object] = {
    "type": "choice",
    "instructions": "Which arXiv category is this paper's primary category?",
    "criteria": CRITERIA,
}

QUESTION_REVERSED: dict[str, object] = {
    "type": "choice",
    "instructions": "Which arXiv category is this paper's primary category?",
    "criteria": CRITERIA_REVERSED,
}

NUM_OPTIONS: int = len(CATEGORIES)  # 8


def collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace (newlines, tabs, multiple spaces) into single spaces."""
    return " ".join(text.split())


def format_state(title: str, abstract: str) -> str:
    """Format paper title and abstract into the exact state string with no leaked metadata."""
    clean_title = collapse_whitespace(title)
    clean_abstract = collapse_whitespace(abstract)
    return f"Title: {clean_title}\n\nAbstract: {clean_abstract}"


def compute_confidences(probabilities: Mapping[str, float], k: int = NUM_OPTIONS) -> tuple[str, float, float]:
    """Return (top_choice, p_top, chance_corrected) from an option->probability mapping.

    Normalizes probabilities if minor floating-point drift exists.
    chance_corrected = (p_top - 1/k) / (1 - 1/k).
    """
    total = sum(float(v) for v in probabilities.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError(f"Invalid probability distribution: {probabilities}")
    norm_probs = {opt: float(val) / total for opt, val in probabilities.items()}
    best_opt = max(norm_probs, key=norm_probs.__getitem__)
    p_top = norm_probs[best_opt]
    chance_corrected = (p_top - 1.0 / k) / (1.0 - 1.0 / k)
    chance_corrected = max(0.0, min(1.0, chance_corrected))
    return best_opt, p_top, chance_corrected
