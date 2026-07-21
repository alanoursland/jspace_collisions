"""J-distance metrics: how close are two J-lens readouts?

The research program (docs/research_program.md, Experiment Family 2) defines
J-distance over either full lens-logit vectors or top-k token rankings:

    - cosine distance between full lens-logit vectors
    - KL / JS divergence over softmaxed lens logits
    - top-k token-set overlap
    - rank-biased overlap of top-k token rankings

All functions are pure NumPy so they can be developed and tested without any
model dependency. Lens logits are 1-D arrays over the vocabulary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EPS = 1e-12


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity. Range [0, 2]; 0 means identical direction."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        raise ValueError("cosine_distance undefined for zero vectors")
    return float(1.0 - np.dot(a, b) / (na * nb))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats over probability vectors."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sum(p * (np.log(p + _EPS) - np.log(q + _EPS))))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric, bounded ([0, ln 2]) divergence — preferred for sweeps."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def topk_indices(logits: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k entries, sorted descending by value."""
    logits = np.asarray(logits)
    idx = np.argpartition(-logits, min(k, logits.size - 1))[:k]
    return idx[np.argsort(-logits[idx], kind="stable")]


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 20) -> float:
    """Jaccard overlap of the top-k index sets of two lens-logit vectors."""
    sa = set(topk_indices(a, k).tolist())
    sb = set(topk_indices(b, k).tolist())
    return len(sa & sb) / len(sa | sb)


def rank_biased_overlap(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    p: float = 0.9,
) -> float:
    """Truncated rank-biased overlap (Webber et al. 2010) of two rankings.

    RBO_trunc = (1 - p) * sum_{d=1..D} p^(d-1) * |A_d ∩ B_d| / d

    where A_d, B_d are the top-d prefixes and D = min(len(a), len(b)).
    Top-weighted: agreement at early ranks counts more. Range [0, 1 - p^D].
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    depth = min(len(ranking_a), len(ranking_b))
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    overlap = 0
    score = 0.0
    for d in range(depth):
        x, y = ranking_a[d], ranking_b[d]
        if x == y:
            overlap += 1
        else:
            overlap += (x in seen_b) + (y in seen_a)
        seen_a.add(x)
        seen_b.add(y)
        score += p**d * overlap / (d + 1)
    return (1.0 - p) * score


def rbo_from_logits(a: np.ndarray, b: np.ndarray, k: int = 50, p: float = 0.9) -> float:
    """Rank-biased overlap of the top-k rankings of two lens-logit vectors."""
    return rank_biased_overlap(topk_indices(a, k).tolist(), topk_indices(b, k).tolist(), p=p)
