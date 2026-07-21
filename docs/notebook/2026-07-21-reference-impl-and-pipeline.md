# 2026-07-21 (later) — Reference implementation in, pipeline validated

## Done

- Obtained the reference implementation via the user's fork
  `alanoursland/jacobian-lens` (commit 581d398). Installed editable; its 32
  tests pass under torch 2.13 / transformers 5.14.
- Read the fitting code: estimator injects one-hot cotangents at every valid
  target position at once (sum over targets, mean over sources), skips the
  first 16 positions (attention sinks) and the final position. Readout is
  `unembed(J_l @ h)` where unembed includes the model's final norm — matches
  the paper formula. `docs/jlens_method.md` updated from the earlier
  paper-only reconstruction.
- Replaced our planned reimplementation with adapters
  (`jspace.lens.adapter.JLensAdapter`, `LogitLensBaseline`) and rebuilt
  `ModelWrapper` on `jlens.from_hf` so layer indexing (block *outputs*),
  encoding, and unembedding are identical to the lens's own.
- Built E9 controls: `ShuffledLens` (vocab permutation), `RandomTransportLens`
  (random orthogonal J_l, Frobenius-matched scale).
- Expanded prompt banks 53 -> 311 pairs.
- Fitting a lens on Qwen2.5-0.5B (CPU): 100 wikitext prompts, all 23 source
  layers, ~4 min/prompt => ~7 h. Checkpoint-per-prompt; interim lenses can be
  snapshotted any time with `scripts/lens_from_checkpoint.py`.
- Validated the E2 sweep end-to-end with a 1-prompt interim lens
  (6 pairs, 3 layers, all 4 lenses): records, manifest, and summary render.

## Findings (smoke-scale, 1-prompt lens — directional only)

- Even with a noisy interim lens, `role_002` ("Alice taught Carol" reversal)
  showed J-distance (JS) 0.039 at L6 with behavior JS 0.177 and an answer
  flip — exactly the H2 collision signature to hunt at scale.
- Qwen2.5-0.5B base is weak on the role-reversal probes (33% both-correct on
  the smoke sample). Collision claims must either condition on both-correct
  pairs or use answer-distribution divergence (which is defined regardless
  of correctness). Larger/instruct models are the Phase-2 confirmation path.

## Methodological note: which control calibrates what

A fixed vocab permutation (ShuffledLens) leaves every *distance* between two
readouts unchanged — distances are permutation-invariant. So:

- Distance-based claims (collision rates) calibrate against
  RandomTransportLens and LogitLensBaseline.
- Content-based claims ("the lens reads out token X") calibrate against
  ShuffledLens.

This should be stated in any writeup; added to docs/metrics.md.

## Next

1. Let the fit finish (or cut at ~50 prompts if quality suffices), save +
   commit the lens artifact.
2. Full 311-pair sweep at layers [4, 8, 12, 16, 20, 22]; analyze; first
   collision report.
3. E1 replication examples from the reference repo's data/ on our fitted
   lens (multi-hop, bug detection, injection recognition).
