# 2026-07-22 — First at-scale collision results (interim lens)

Setup: Qwen2.5-0.5B, interim J-lens (45 wikitext prompts — final is 100),
all 311 benchmark pairs, layers {4, 8, 12, 16, 20, 22}, four lenses
(jlens, shuffled, random-transport, logit-lens). 7,464 records.
Full tables: `results/e02_collision_search/sweep_interim50/summary.md`.

## Headline: H2-style collisions exist, at scale, in the workable category

The cleanest examples are role reversals where the model *demonstrably
performs the task* (both variants answered correctly, confidently) while
mid-layer J-lens readouts are nearly identical:

| pair | prompts | L16 J-dist (JS) | top-20 Jaccard | behavior |
|---|---|---|---|---|
| role_025 | "Alice praised David." / reversed | 0.0034 | 0.82 | P(correct)≈.99/.98, flip |
| role_026 | "Alice blamed David." / reversed | 0.0017 | 0.82 | .97/.97, flip |
| role_015 | "Alice praised Carol." / reversed | 0.0068 | 0.74 | .97/.96, flip |
| role_035 | "Alice praised Emma." / reversed | 0.0187 | 0.82 | .99/.95, flip |

Reading: at L16 the two role-reversed states are essentially the same point
in J-space (JS divergence of readout distributions ~0.002–0.02, i.e. within
noise of identical; top-20 tokens overlap >80%) while downstream behavior is
close to maximally different (the model answers the probe correctly, in
opposite directions, with ~97–99% confidence). The bag-of-tokens readout
contains {Alice, David, praised, ...} in both cases; who-did-what-to-whom is
not in the readout at that layer.

## Calibration and context

- Mean same-pair J-distance (jlens) rises with depth: 0.074 (L4) → 0.182
  (L20). The *random-transport control* stays flat ~0.05–0.09. So the fitted
  lens is on average MORE discriminative between pair variants than a random
  transport — the near-zero collisions above are not "the lens can't see
  anything"; the lens sees plenty on average and still sees nothing here.
- Shuffled-lens rows are identical to jlens rows, as predicted (permutation
  invariance of distances) — recorded as a consistency check, not evidence.
- Layer pattern: collision candidates concentrate at L12–L16; by L20–22
  J-distances for behavior-divergent pairs roughly double (0.07→0.19 for
  role reversal) — late layers must encode the answer they are about to
  produce. Polysemy shows this most sharply: behavior-divergent polysemy
  pairs jump from J-dist ~0.12–0.19 (L4–L16) to 0.55–0.65 (L20–22) — sense
  resolution becomes readable only near the output.

## Competence gradient (0.5B limitation, as expected)

both-correct rates: role_reversal 67%, relation_binding 33%, polysemy 21%,
negation/causal_flip/safety_latent 0%. The last three categories produce
~zero behavior distance at this scale — the model can't do those probe
tasks, so they are uninformative here and need a larger/instruct model
(Phase 2 GPU work). All quantitative collision claims are therefore
currently restricted to role_reversal (and partially relation_binding).

## Caveats before anyone gets excited

1. Interim lens (45 prompts). Final 100-prompt rerun + convergence
   comparison pending (fit at 62/100).
2. Readout at the final prompt token only; the binding information could be
   readable at other positions (e.g., at the name tokens). Position sweep is
   a needed robustness check before the claim hardens.
3. No patch confirmation yet: we have same-readout/different-behavior, not
   yet "behavior follows the non-J-space state under patching." That is the
   crux experiment (E2 step 5) and remains open.
4. Small base model; the paper's claims target much larger models.

## Next

- E1 lens-quality evals running (multihop, association, typo) — sanity check
  that the interim lens reads out anything at all vs controls.
- Final-lens rerun + convergence check when fit completes.
- Position-sweep robustness check and patch confirmation are the next new
  experiments to build.
