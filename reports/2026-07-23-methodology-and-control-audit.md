# Lab Report: Methodology and Control Audit

Date: 2026-07-23  
Scope: E1/E2 metrics, causal criteria, precision, and null controls  
Result classification: two material corrections to the original analysis

## Purpose

Audit whether the collision-search procedure distinguishes a globally hidden
state from a readout artifact, model incompetence, lens noise, or an
inapplicable null control.

## Correction 1: final-token distance is not global distance

The original E2 sweep compared readouts at the final statement token. Patch
confirmation showed that this token could be causally inert even when the
prompt pair produced different behavior.

The revised sweep retains the legacy final-token measurements and adds:

- mean aligned-position JS;
- maximum aligned-position JS (`scan_js`);
- bag-of-positions JS;
- mean and minimum aligned top-k overlap;
- token counts to identify unaligned pairs.

For monitoring claims, `scan_js` is the strict measure: a pair is not globally
close when any inspected aligned position clearly separates it.

This correction eliminated all five legacy 0.5B collisions at L16.

## Correction 2: shuffled vocabulary is not a distance control

`ShuffledLens` applies one fixed vocabulary permutation to both readouts. For
pairwise metrics:

```text
d(Pp, Qp) = d(P, Q)
```

for a common permutation `p`, when `d` is JS divergence, cosine distance, or
set/rank overlap with both indices permuted together.

Consequences:

- shuffled-token labels remain a valid E1 control for claims that a readout
  names the correct token;
- they are mathematically vacuous for E2 pairwise-distance claims;
- future E2 runs exclude this control;
- historical shuffled-distance records are retained for provenance but
  omitted from control summaries.

Applicable E2 controls are:

- random orthogonal transport, Frobenius-matched to the fitted Jacobian;
- the identity/logit lens.

A regression test now locks in the permutation-invariance result.

## Causal evidence standard

A strong collision candidate should satisfy:

1. both variants are answered correctly;
2. behavior differs materially;
3. the proposed readout remains unusually close under position-aware metrics;
4. it separates from applicable distance controls;
5. patching the proposed site transfers behavior;
6. the result is stable across lens convergence, precision, paraphrases, and
   seeds.

The completed experiments show why each criterion matters:

- `role_025` at 0.5B fails criteria 3 and 5 at the final token.
- `role_130` at 1.5B passes criteria 1-5 and precision stability, but still
  needs paraphrase/seed stability and is threshold-sensitive.
- the 3B candidates do not pass criterion 3.

## Precision audit

One-prompt BF16 Jacobians varied by up to approximately 4% at early layers
when `dim_batch` changed. Production 3B fits therefore use one fixed
`dim_batch=8`.

After 100-prompt averaging, the matched 1.5B BF16/FP32 transport difference is
below 1% at every fitted layer. E1 scores are effectively identical, and the
`role_130` causal near-collision survives.

Model inference behavior is more precision-sensitive than the fitted
transport: 1.5B role-reversal both-correct competence changes from 87% in FP32
to 47% in BF16. Comparisons must therefore record both lens-fitting precision
and evaluation-model precision.

## Reporting rules adopted

- Report continuous distances and control ratios, not only threshold counts.
- Separate final-token, all-position, and bag-of-positions claims.
- Report competence by category before collision rates.
- Use “near-collision” when a candidate is unusually close but misses the
  declared strict threshold.
- Treat visible/fiber decompositions as curves over singular rank or mass,
  not as one binary 90% split.
- Store model revision, precision, fit settings, code revision, and artifact
  SHA-256 in manifests.
- Keep large `.pt` files outside ordinary Git history.

## Evidence

- [`docs/metrics.md`](../docs/metrics.md)
- [`experiments/e02_collision_search/README.md`](../experiments/e02_collision_search/README.md)
- [`tests/test_jdist.py`](../tests/test_jdist.py)
- [`docs/notebook/2026-07-23-provenance-and-gpu-handoff.md`](../docs/notebook/2026-07-23-provenance-and-gpu-handoff.md)

