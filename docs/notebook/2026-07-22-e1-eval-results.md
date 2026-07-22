# 2026-07-22 — E1 lens-quality evals (interim lens): replication passes

Paper eval sets from the reference repo, interim 45-prompt lens, all 23
layers, pass@k = fraction of latent intermediates at lens rank <= k at the
pre-target position (min over layers).
Full table: `results/e01_replication/evals_interim50/summary.md`.

| eval | jlens p@1 | jlens p@10 | shuffled | random-transport | logit-lens p@10 |
|---|---|---|---|---|---|
| multihop | 0.34 | 0.47 | 0.00 | 0.00–0.09 | 0.25 |
| typo | 0.18 | 0.43 | 0.00 | 0.00 | 0.09 |
| association | 0.00 | 0.00 | 0.00 | 0.00–0.01 | 0.00 |

## Reading

1. **The core paper claim replicates on a 0.5B open model with a 45-prompt
   CPU-fitted lens**: latent multi-hop intermediates (e.g. "Brazil" in
   "the country where Carnival is celebrated") surface at lens rank 1 in a
   third of items, with shuffled and random-transport controls at ~zero.
   This is a content claim, so ShuffledLens is the operative control
   (docs/metrics.md) — separation is decisive.
2. **J-lens > logit lens by ~2x** on both informative evals — the fitted
   transport adds real information beyond the unembedding alone.
3. **association is a null across all lenses** — uninformative at this
   model scale (either the associations aren't represented or our readout
   position convention mismatches that eval's design; check against the
   repo's per-eval conventions before reading anything into it).

## Why this matters for the collision story

The E2 finding is now two-sided: the *same* lens that demonstrably reads
out latent verbalizable content (E1, far above all controls) is *unable*
to distinguish role-reversed bindings the model itself acts on correctly
(E2, L16 readouts near-identical). "The lens works" and "the lens misses
binding" are simultaneously true — which is precisely H1+H2.

Pending: rerun with the final 100-prompt lens (fit at 65/100) for the
convergence check.
