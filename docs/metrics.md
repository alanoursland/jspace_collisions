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

Defaults (k, p, temperature) are starting points; sensitivity to them is part
of Phase 1 calibration and any change must be recorded here.

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

Every readout metric is reported alongside the same metric computed with a
shuffled-label lens and a random-orthogonal-transport lens. A finding requires
separation from both.
