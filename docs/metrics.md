# Metric Definitions

Canonical definitions for metrics used across experiments. Every metric
reported in a writeup must exist as a tested function in `src/jspace/metrics/`
and be defined here. Program-level metric lists: `docs/research_program.md`,
"Key Metrics".

## Notation

- `h` — residual-stream activation at a layer and position
- `L(h)` — J-lens readout (lens logits over the vocabulary)
- `B(h)` — behavioral measurement after patching h into a run
- `A(h)` — auxiliary internal state (SAE features, attention, etc.)

## J-distance (readout similarity) — `jspace.metrics.jdist`

| Metric | Function | Notes |
|---|---|---|
| Cosine distance | `cosine_distance(L(h1), L(h2))` | on raw lens logits; range [0, 2] |
| KL divergence | `kl_divergence(softmax(L1), softmax(L2))` | asymmetric; report direction |
| JS divergence | `js_divergence(...)` | symmetric, bounded [0, ln 2]; default for sweeps |
| Top-k overlap | `topk_overlap(L1, L2, k)` | Jaccard of top-k index sets; k=20 default |
| Rank-biased overlap | `rbo_from_logits(L1, L2, k, p)` | truncated RBO, Webber et al. 2010; top-weighted; k=50, p=0.9 default |
| Position-aware distances | `position_readout_distances(L_seq1, L_seq2, k)` | aligned positions; reports final/mean/max-scan JS, order-free bag JS, and mean/min top-k overlap |

Defaults (k, p, temperature) are starting points; sensitivity to them is part
of Phase 1 calibration and any change must be recorded here.

For equal-length prompt pairs, `scan_js` is the strict monitoring distance:
the maximum JS divergence at any aligned position. A low final-token distance
is not an all-position collision when `scan_js` is high. `bag_js` averages
the per-position distributions before comparison and therefore tests an
order-free bag-of-readouts monitor; it must not be interpreted as preserving
binding or position.

## Behavior distance — `jspace.metrics.behavior`

| Metric | Function | Notes |
|---|---|---|
| Answer mismatch | `not answer_match(a1, a2)` | normalized exact match |
| Output distribution divergence | `output_distribution_divergence(p, q)` | JS over next-token or answer-choice distributions |
| Intervention effect size | `cohens_h(p1, p2)` | difference of proportions (e.g. flip rates) |

## Search heuristics

`collision_score(j_dist, b_dist)` ranks candidate collisions
(high behavior-distance at low J-distance). Heuristic only — never a reported
result; reported collisions use thresholded J-distance + behavior-distance
with their null-control baselines.

## Causal metrics (Phase 1+, model-dependent)

Defined in the program; implementations land with the patching harness:
swap success rate, ablation damage, mediated-effect fraction through J-space,
patch transfer success, fiber behavior variance
`Var(B(h+delta) | L(h+delta) ~= L(h))`.

## Null baselines (E9)

Every readout metric is reported alongside its null-control baseline — but
the *right* control depends on the claim:

- **Distance claims** (collision rates, J-distance thresholds): a fixed vocab
  permutation leaves all pairwise distances unchanged, so ShuffledLens is
  vacuous here. Calibrate against RandomTransportLens (random orthogonal J_l,
  Frobenius-matched) and LogitLensBaseline (identity transport).
- **Content claims** ("the lens reads out token X"): calibrate against
  ShuffledLens; a shuffled lens recovering "meaningful" tokens above chance
  flags over-interpretation.

A finding requires separation from the applicable controls.
