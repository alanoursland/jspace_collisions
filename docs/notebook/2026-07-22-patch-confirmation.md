# 2026-07-22 — Patch confirmation: colliding site is not the causal site

E2 step 5 on the top 8 patch-eligible collision pairs (L16, both-correct,
flipped answers; role_026/035/125/015/032/021 survived the equal-length
filter). Conditions patch variant A's L16 residuals into B's probed run and
vice versa (16 direction-runs per condition).
Code: `experiments/e02_collision_search/patch_confirm.py`.
Results: `results/e02_collision_search/patch_confirm_v1/`.

| condition | mean behavior transfer | flips to source answer |
|---|---|---|
| full swap, final statement token only | 0.00 | 0/16 |
| full swap, all statement tokens | 1.00 | 16/16 |
| lens-visible component only (rank-367/896, 90% mass) | 0.73 | 10/16 |
| fiber (low-gain) component only | 0.64 | 7/16 |

## The key negative result

The site where the collision lives — the statement-final token at L16,
where readouts are near-identical while behavior flips — is **causally
inert for the probe behavior**: swapping it entirely moves nothing (0/16).
The behavior is carried by the *name-position* residuals (full-statement
swap transfers 16/16), and at those positions the per-position readouts
differ trivially — position 0 reads "Alice" in one variant and "David" in
the other. A monitor that scans **all positions** sees the difference;
only a monitor reading the statement-final "summary" position is blind.

So the strong causal claim — "behavior is carried by J-space-invisible
state" — is NOT supported at this site. What survives is weaker but real:

1. At L16 the model has not composed a bound proposition at the
   statement-final token (or the probe doesn't consult it); the probe
   plausibly attends back to name positions directly. Binding here is
   carried *positionally*, not in any single readout vector.
2. Same-readout-different-behavior at a single site is a monitoring
   caveat, not evidence of a global J-space fiber: you must check the
   site's causal relevance before interpreting its readout.

## What stays interesting

- The jvis/fiber split at the causal (name) positions is not clean: the
  lens-visible component alone transfers 0.73 and flips 10/16, but the
  low-gain "fiber" component alone still transfers 0.64 and flips 7/16
  (components interact nonlinearly; transfers need not sum to 1). A
  substantial share of behaviorally-relevant information rides directions
  the lens transport attenuates. This is worth a cleaner follow-up with
  per-position readout distances and varying the mass threshold.
- Methodological upgrade for the collision search: J-distance should be
  measured at causally sufficient sites, or over position-aggregated
  readouts (the bag of per-position top tokens), not at one conventional
  position. The current benchmark's "collisions" are real observations
  attached to the wrong site.

## Revised program status

H2 as originally imagined (binding invisible to J-space, causally) is
not yet demonstrated at 0.5B — our cleanest collisions turned out to be
epiphenomenal at their site. The refined question, which the harness can
now ask directly: is there any (site, aggregation) under which the
causally sufficient difference is lens-invisible? Candidates: positions
where an aggregated proposition must exist (multi-hop intermediates,
pronoun resolution sites), and tasks where the probe cannot re-derive the
answer from surface positions.

This is the adversarial step doing its job: the pipeline caught our own
over-interpretation before it hardened into a claim.
