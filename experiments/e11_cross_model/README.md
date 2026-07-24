# Cross-Model and Training-Stage Comparison

Program family: E11 | Plan: Phase 6 | Status: initial size axis complete

Run the battery across model families, sizes, and base-vs-instruct
checkpoints. Deliverable: a cross-model J-space globality leaderboard with
capability and failure metrics.

The first matched-family size pass covers Qwen2.5 base models at 0.5B, 1.5B,
and 3B. The 1.5B and 3B lenses use normalized seven-layer grids and
20/50/100-prompt convergence snapshots. Results and caveats are recorded in
`docs/notebook/2026-07-23-provenance-and-gpu-handoff.md`.

The 1.5B BF16 precision control is complete. Next, add the base-vs-instruct
axis to separate model competence from representation faithfulness.

See `docs/research_program.md` for the full procedure.
