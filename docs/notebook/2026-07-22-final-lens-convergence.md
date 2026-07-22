# 2026-07-22 — Final 100-prompt lens: results converge with interim

The 100-prompt lens finished (3.6h CPU, 3 container restarts survived via
checkpointing; artifact committed at `data/lens/qwen2.5-0.5b_wikitext100.pt`).
Full 311-pair sweep rerun with it: `results/e02_collision_search/sweep_final/`.

## Convergence check: 45-prompt vs 100-prompt lens

| quantity | interim (45) | final (100) |
|---|---|---|
| mean J-dist (jlens) L4..L22 | .074 .088 .129 .115 .182 .170 | .073 .084 .130 .104 .178 .164 |
| top collision pair set | role_{025,026,035,015,071,125} | identical |
| role_025 @L16 (J-dist / top-20 Jaccard) | 0.0034 / 0.82 | 0.003 / 0.905 |
| role_026 @L16 | 0.0017 / 0.82 | 0.001 / 0.905 |
| role_015 @L16 | 0.0068 / 0.74 | 0.005 / 0.74 |

Reading: doubling the fitting corpus changes mean J-distances by <=0.011 and
leaves the collision ranking intact; the strongest collisions become slightly
*cleaner* (higher readout overlap at the same near-zero distance). The
collisions are stable properties of the (model, lens-method) pair, not
artifacts of an under-fitted lens. This matches the paper's note that lens
quality saturates quickly with corpus size, and satisfies the Phase-1
"stable readout metrics" exit criterion for this axis (paraphrase/seed
stability still to be run).

Layer pattern confirmed on the final lens: role-reversal behavior-divergent
pairs dip to J-dist 0.060 at L16 (the collision band) then double to ~0.19
by L20; polysemy sense-resolution stays unreadable until L20-22 (0.07 at
L16 -> 0.55-0.64 at L20-22).

Final E1 evals (pass@k vs controls with the 100-prompt lens) running;
results appended to `results/e01_replication/evals_final/` when done.
