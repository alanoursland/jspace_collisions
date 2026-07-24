# J-Space Collision Search

Program family: E2 | Plan: Phase 2 | Status: initial sweep and causal audit complete; revised search in progress

Find prompt/activation pairs with near-identical J-lens readouts but
different behavior. Uses `jspace.prompts` minimal pairs, `jspace.metrics`
J-distance and behavior-distance, ranked by `collision_score`. Top candidates
are confirmed by cross-patching activations within each pair.

Primary outcome: collision rate by task class, layer, position. This family
is the core of the first concrete study ("Hidden Folds in J-Space").

The first final-token sweep found strong-looking role-reversal collisions, but
patch confirmation showed that the colliding final site was causally inert.
The sweep now records aligned all-position metrics; a causal collision claim
must remain close under that scan and transfer behavior when patched.

Distance calibration uses the random-transport and logit-lens controls. A
shared shuffled-vocabulary lens is intentionally excluded because pairwise
JS, cosine, and overlap distances are invariant to a common label
permutation; that control remains appropriate for E1 token-identity claims.

See `docs/research_program.md` for the full procedure.
