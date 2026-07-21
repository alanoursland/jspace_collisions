"""Behavior-distance metrics: how different is downstream behavior?

B(h) in the research program: final answers, output distributions, task
scores, or safety-classifier categories measured after patching a state into
a run. These metrics compare two such measurements.
"""

from __future__ import annotations

import numpy as np

from jspace.metrics.jdist import js_divergence


def answer_match(answer_a: str, answer_b: str, *, normalize: bool = True) -> bool:
    """Exact answer match, optionally case/whitespace-normalized."""
    if normalize:
        answer_a = " ".join(answer_a.lower().split())
        answer_b = " ".join(answer_b.lower().split())
    return answer_a == answer_b


def output_distribution_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS divergence between two next-token (or answer-choice) distributions."""
    return js_divergence(p, q)


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for a difference between two proportions.

    Used for intervention effect sizes, e.g. P(answer flips) under a patch vs
    under a control patch. |h| ~ 0.2 small, 0.5 medium, 0.8 large.
    """
    for p in (p1, p2):
        if not 0.0 <= p <= 1.0:
            raise ValueError("proportions must be in [0, 1]")
    return 2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def collision_score(j_distance: float, behavior_distance: float, j_scale: float = 1.0) -> float:
    """Rank candidate collisions: high behavior-distance at low J-distance.

    score = behavior_distance / (1 + j_distance / j_scale)

    Monotone up in behavior-distance, down in J-distance; j_scale sets the
    J-distance at which the penalty reaches 2x. This is a search heuristic for
    ranking candidate pairs, not a reported metric.
    """
    if j_distance < 0 or behavior_distance < 0:
        raise ValueError("distances must be non-negative")
    return behavior_distance / (1.0 + j_distance / j_scale)
