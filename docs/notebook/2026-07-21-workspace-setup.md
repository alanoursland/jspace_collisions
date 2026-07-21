# 2026-07-21 — Workspace setup

## Done

- Scaffolded the repository: package layout, pyproject (hatchling, uv-friendly),
  pytest suite, per-experiment directories e01–e12 with scoped READMEs.
- Committed the research program verbatim to `docs/research_program.md`;
  operational mapping in `PLAN.md`.
- Implemented and tested the pure-NumPy layer:
  - J-distance metrics (cosine, KL/JS, top-k Jaccard, truncated RBO)
  - behavior metrics (answer match, JS divergence, Cohen's h) and the
    `collision_score` search heuristic
  - minimal-pair prompt banks (~45 starter pairs across role reversal,
    relation binding, negation, causal flips, polysemy, safety-latent)
  - patching reference semantics (`apply_patch_array`) that the future
    model harness must match exactly
- Defined `Lens`/`Readout` interfaces from the paper's description.

## Decisions

- NumPy at all interface boundaries; torch confined behind the `[models]`
  extra so the benchmark/metric layer runs anywhere.
- `data/` and `results/` are manifest-only in git; payloads gitignored.
- Null controls (E9) treated as a standing obligation attached to every
  readout claim, not a standalone phase.

## Open questions / blockers

- Access to `anthropics/jacobian-lens`: not reachable from this session
  (GitHub scope limited to this repo). Options: add as session source,
  vendor a pinned copy, or reimplement from the paper. Interfaces in
  `src/jspace/lens/interface.py` are provisional until reconciled.
- GPU environment for Phase 1 lens fitting is undecided; this container is
  CPU-only.
- Primary model choice defaults to the smallest Qwen the reference
  implementation supports; confirm once the implementation is in hand.
