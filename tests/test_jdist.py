import numpy as np
import pytest

from jspace.metrics import (
    cosine_distance,
    js_divergence,
    kl_divergence,
    position_readout_distances,
    rank_biased_overlap,
    softmax,
    topk_overlap,
)
from jspace.metrics.jdist import rbo_from_logits, topk_indices


def test_softmax_normalizes():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert p[2] > p[1] > p[0]


def test_cosine_distance_bounds():
    a = np.array([1.0, 0.0])
    assert cosine_distance(a, a) == pytest.approx(0.0)
    assert cosine_distance(a, np.array([0.0, 1.0])) == pytest.approx(1.0)
    assert cosine_distance(a, -a) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        cosine_distance(a, np.zeros(2))


def test_kl_and_js():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    q = softmax(np.array([3.0, 2.0, 1.0]))
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    assert kl_divergence(p, q) > 0
    assert js_divergence(p, q) == pytest.approx(js_divergence(q, p))
    assert 0 <= js_divergence(p, q) <= np.log(2) + 1e-9


def test_js_roundoff_never_returns_negative():
    p = np.array([1.0 - 3e-15, 1e-15, 1e-15, 1e-15])
    q = p.copy()
    assert js_divergence(p, q) >= 0.0


def test_topk_indices_sorted_desc():
    logits = np.array([0.1, 5.0, 3.0, 4.0, 0.2])
    assert topk_indices(logits, 3).tolist() == [1, 3, 2]


def test_topk_overlap():
    a = np.array([5.0, 4.0, 3.0, 0.0, 0.0])
    b = np.array([0.0, 4.0, 3.0, 5.0, 0.0])
    # top-3 sets: {0,1,2} vs {1,2,3} -> jaccard 2/4
    assert topk_overlap(a, b, k=3) == pytest.approx(0.5)
    assert topk_overlap(a, a, k=3) == pytest.approx(1.0)


def test_position_readout_distances_identical():
    logits = np.array([[5.0, 1.0, 0.0], [0.0, 2.0, 4.0]])
    distances = position_readout_distances(logits, logits, k=2)
    assert distances["final_js"] == pytest.approx(0.0)
    assert distances["mean_js"] == pytest.approx(0.0)
    assert distances["scan_js"] == pytest.approx(0.0)
    assert distances["bag_js"] == pytest.approx(0.0)
    assert distances["mean_topk_overlap"] == pytest.approx(1.0)
    assert distances["min_topk_overlap"] == pytest.approx(1.0)


def test_position_readout_distances_detects_one_visible_site():
    a = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    b = np.array([[0.0, 0.0, 8.0], [0.0, 8.0, 0.0]])
    distances = position_readout_distances(a, b, k=1)
    assert distances["final_js"] == pytest.approx(0.0)
    assert distances["scan_js"] > distances["mean_js"] > 0.0
    assert distances["min_topk_overlap"] == pytest.approx(0.0)
    assert distances["mean_topk_overlap"] == pytest.approx(0.5)


def test_position_readout_distances_rejects_unaligned_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        position_readout_distances(np.zeros((2, 3)), np.zeros((3, 3)))


def test_shared_vocab_permutation_is_not_a_distance_control():
    a = np.array([5.0, 1.0, 3.0, -2.0, 0.0])
    b = np.array([0.0, 4.0, 2.0, -1.0, 3.0])
    permutation = np.array([2, 4, 0, 3, 1])

    assert cosine_distance(a, b) == pytest.approx(
        cosine_distance(a[permutation], b[permutation])
    )
    assert js_divergence(softmax(a), softmax(b)) == pytest.approx(
        js_divergence(softmax(a[permutation]), softmax(b[permutation]))
    )
    assert topk_overlap(a, b, k=3) == pytest.approx(
        topk_overlap(a[permutation], b[permutation], k=3)
    )


def test_rbo_identical_and_disjoint():
    assert rank_biased_overlap([1, 2, 3], [1, 2, 3], p=0.9) == pytest.approx(
        (1 - 0.9) * (1 + 0.9 + 0.9**2)
    )
    assert rank_biased_overlap([1, 2, 3], [4, 5, 6], p=0.9) == pytest.approx(0.0)


def test_rbo_top_weighted():
    # agreement at rank 1 beats agreement at rank 3
    early = rank_biased_overlap([1, 8, 9], [1, 6, 7], p=0.9)
    late = rank_biased_overlap([8, 9, 1], [6, 7, 1], p=0.9)
    assert early > late


def test_rbo_from_logits_self():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    self_rbo = rbo_from_logits(a, a, k=10)
    other_rbo = rbo_from_logits(a, rng.normal(size=100), k=10)
    assert self_rbo > other_rbo
