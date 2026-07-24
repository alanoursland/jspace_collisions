# Cross-Model and Training-Stage Comparison

Program family: E11 | Plan: Phase 6 | Status: size and training-stage controls complete

Run the battery across model families, sizes, and base-vs-instruct
checkpoints. Deliverable: a cross-model J-space globality leaderboard with
capability and failure metrics.

The first matched-family size pass covers Qwen2.5 base models at 0.5B, 1.5B,
and 3B. The 1.5B and 3B lenses use normalized seven-layer grids and
20/50/100-prompt convergence snapshots. Results and caveats are recorded in
`docs/notebook/2026-07-23-provenance-and-gpu-handoff.md`.

The 1.5B BF16 precision control and matched 3B base-vs-instruct axis are
complete. The Instruct checkpoint restored much of the raw-prompt behavioral
competence but still produced no strict all-position collision at L25.

## 3B-Instruct control

Use the immutable `Qwen/Qwen2.5-3B-Instruct` revision recorded in the lens
manifest. Evaluate two prompt conditions before interpreting lens geometry:

1. `raw`: identical prompt strings and continuation scoring used for the base
   model, isolating the training-stage/checkpoint change.
2. `chat`: the tokenizer-native user/assistant template, testing whether the
   base model's weak role/binding competence is an answer-format artifact.

The behavior-only gate is `run_competence.py`. The checkpoint passed that gate
under raw prompting, so a 100-prompt BF16 lens was fitted at normalized layers
`[5, 10, 15, 20, 25, 30, 34]` with sequence length 96 and `dim_batch=8`.
Raw and chat E1/E2 outputs were kept separate.

At L25, raw prompting produced 160 unique both-correct answer flips and 27
legacy final-token hits, but zero strict all-position hits. Chat prompting
produced 48 competent flips and 22 legacy hits, also with zero strict hits.
The converged raw nearest candidate had scan JS .301; the nearest chat record
had final-token JS .007 but scan JS .612. No patch confirmation was warranted.

Full results, competence rates, convergence, controls, and limitations are in
`reports/2026-07-23-3b-instruct-control.md`.

See `docs/research_program.md` for the full procedure.
