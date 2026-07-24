# Execution Plan

Operational plan for the research program in `docs/research_program.md`.
The program defines twelve experiment families (E1–E12) and a six-month
milestone arc; this document maps that onto concrete phases, deliverables, and
repository artifacts, and records current status.

## Ground rules

- Every experiment family gets a directory `experiments/eNN_*` with its own
  README (goal, procedure, status) and runnable scripts that import shared code
  from `src/jspace/`. No experiment-local copies of metrics or harness code.
- All metrics used in papers/reports must exist as tested functions in
  `src/jspace/metrics/` before results are reported with them.
- Null controls (E9) run alongside every readout claim, not as a final phase —
  any "the lens shows X" result is paired with its shuffled-lens baseline.
- Findings, dead ends, and decisions go in `docs/notebook/` as dated entries.
- Heavy artifacts (activations, model weights) never enter git; `data/` and
  `results/` carry manifests only.

## Compute posture

The local workstation has an RTX 3080 Ti (12 GiB VRAM) and 96 GiB CPU RAM.
The 0.5B CPU artifacts remain the historical baseline; matched 1.5B and 3B
lenses now run locally on CUDA. A seven-layer, sequence-length-96 3B BF16 fit
uses about 8.4 GiB and 28 seconds per prompt at `dim_batch=8`. Ordinary 7B
retained-gradient fitting is still outside this VRAM budget without a
validated offload or recomputation method.

## Phase 0 — Workspace setup (now)

- [x] Repository scaffold, package layout, pyproject, test harness
- [x] Research program committed to `docs/research_program.md`
- [x] J-distance metrics implemented and tested (`src/jspace/metrics/jdist.py`):
      cosine distance on lens logits, KL/JS on softmaxed logits, top-k overlap,
      rank-biased overlap
- [x] Behavior-distance metrics implemented and tested
      (`src/jspace/metrics/behavior.py`): answer match, output-distribution
      divergence, effect-size helpers
- [x] Minimal-pair prompt generators with starter banks
      (`src/jspace/prompts/`): role reversal, negation, causal flips,
      polysemy, relation/binding, safety-relevant latent states
- [x] Lens and patching interfaces defined (`src/jspace/lens/`,
      `src/jspace/patching/`) as typed protocols so experiment code can be
      written before the reference implementation is integrated
- [x] Obtain and pin the Anthropic reference implementation — via the user's
      fork `alanoursland/jacobian-lens` (commit 581d398), installed editable;
      all 32 of its tests pass in this environment. Our lens layer is now an
      adapter over `jlens.JacobianLens` (`src/jspace/lens/adapter.py`), not a
      reimplementation. Method notes: `docs/jlens_method.md`.

## Phase 1 — Baseline / replication (program Month 1; E1, E9 controls)

Goal: a working J-lens on at least one open model, with trustworthy metrics.

Status 2026-07-22: substantially COMPLETE on CPU at 0.5B scale.
- Lens fitted (100 wikitext prompts, committed: `data/lens/`), convergence
  vs a 45-prompt interim lens verified (notebook 2026-07-22).
- E1 replication: multihop pass@1 0.34 / typo 0.18 vs 0.00 shuffled
  control — the paper's core readout claim replicates on Qwen2.5-0.5B.
- E2 full sweep run twice (interim + final lens): stable role-reversal
  collisions found at L12-16 (J-dist ~0.001-0.02, top-20 Jaccard up to
  0.9, behavior JS >0.5 with confident correct-but-opposite answers).
- Patch confirmation completed: the strongest final-token collision site was
  causally inert, while swapping all statement positions transferred behavior
  completely. The strong hidden-fiber claim is not established at 0.5B.
- A position-aware regression check now makes the distinction explicit:
  `role_025` at L16 has final-token JS 0.00275 but max-position scan JS 0.64783.
- Remaining for Phase 1/2 exit: paraphrase/seed stability, causally gated
  position-aware collision search, and the matched larger-model GPU series.

1. Integrate the reference implementation behind `jspace.lens.JLens`
   (adapter conforming to `lens/interface.py`).
2. Reproduce the paper's qualitative examples on a small Qwen-family model:
   multi-hop latent concepts, silent arithmetic, bug detection,
   prompt-injection recognition, concept swaps.
3. Build the activation recording pipeline: for each (prompt, layer, position)
   store residual activation, lens logits, top-k tokens, model output
   distribution. Format: safetensors + JSON manifest per run (see
   `data/README.md`).
4. Stand up the patching harness (`patching/harness.py`) for replacement,
   addition/subtraction, projection, and nullspace-constrained perturbation.
5. Fit the first null-control lenses (shuffled labels, random orthogonal
   transport) and record baseline false-positive rates.

Exit criteria: paper examples reproduced qualitatively; readout metrics stable
across seeds and paraphrases; first activation dataset with manifest; control
lens baseline numbers recorded in the notebook.

## Phase 2 — Collision search (Month 2; E2)

Goal: first same-J, different-behavior pairs. This feeds the "first concrete
study" (Hidden Folds in J-Space) directly.

1. Expand prompt banks to the target 500 minimal pairs across the five
   categories (role reversal, negation, polysemy, multi-token concepts,
   safety-relevant recognition). Generators exist; this phase is bank curation
   plus behavioral probe questions with graded answers.
2. Run the sweep: J-distance and behavior-distance for every pair at each
   candidate workspace layer; emit a ranked collision table
   (low J-distance x high behavior-distance).
3. Confirm top candidates causally: cross-patch activations within each pair
   and record whether behavior follows the patched J-space content or the
   non-J-space context.
4. Report collision rate by task class, layer, position.

Exit criteria: reproducible collision table with confidence tiers; at least a
first set of patch-confirmed collisions (or a documented negative result).

## Phase 3 — Fiber mapping (Month 3; E3)

1. Nullspace sampler: perturbations that hold L(h) approximately fixed
   (random under norm constraint, then gradient-based behavior maximization
   subject to J-distance <= epsilon).
2. Estimate `fiber_behavior_variance` across layers and tasks.
3. Identify layers where J-space is most/least behaviorally sufficient.

## Phase 4 — Binding, computation-splits, and polysemy (Month 4; E4, E5, E6)

1. Binding benchmark (E5): minimal pairs with identical entity sets, different
   role bindings; readout comparison + entity/relation swap interventions.
2. Same-J different-computation tasks (E4): report-vs-use task pairs; taxonomy
   of operations that consult J-space vs bypass it.
3. Polysemy suite (E6): sense-conflation tests for bank/charge/bug/plant/
   draft/proof; test context-specific sense separation.

## Phase 5 — Safety monitor stress tests (Month 5; E10, plus E7 coverage)

1. Multi-token/abstract concept coverage benchmark (E7) — needed first, since
   monitor evasion often routes through concepts the token lens can't name.
2. Controlled, benign prompts eliciting safety-relevant latent states
   (evaluation awareness, deception recognition, injection recognition).
3. Measure `monitor_coverage` and `evasion_gap` for token-level J-space
   monitors vs richer monitors (phrase lens, SAE features, supervised probes).

## Phase 6 — Synthesis and improved lenses (Month 6; E8, E11, E12)

1. Context-specific Jacobians (E8): domain-fitted lenses vs the global lens on
   collision-heavy domains.
2. Cross-model battery (E11) on 2–3 model families / sizes; globality
   leaderboard.
3. Prototype one or two improved lenses (E12) — most likely a relation lens
   and a fiber-aware lens — and re-run the hardest failure cases.
4. Final report: benchmarks, activation traces, evaluation harness, and
   calibrated interpretation standards.

## First concrete study (fast-tracked)

"Hidden Folds in J-Space: Same-Readout, Different-Behavior Tests for the
Jacobian Lens" = Phase 1 + Phase 2 only. Everything in phases 0–2 is scoped so
this study is publishable even if the rest of the program shifts.

## Immediate next actions

1. Add a Qwen2.5-3B-Instruct competence/readout axis; the 3B base model is weak
   on the current role/binding answer format despite stronger multihop E1
   readout.
2. Expand paraphrase/seed coverage around the 1.5B `role_130` near-collision.
3. Replace binary collision counts with continuous scan-distance versus
   behavior-distance curves calibrated against controls.
4. Causally patch only candidates that remain unusually close under the
   all-position scan metric.
5. Evaluate whether activation or gradient checkpointing can extend the same
   estimator beyond 3B without changing its mathematical target.

Detailed gates and artifact conventions: `docs/gpu_campaign.md`.

## Risk register

- Reference implementation API may not match our interfaces → adapters are
  thin by design; reconcile in Phase 1 before any experiment code depends on
  details.
- Reference implementation may not be public/accessible → fallback: implement
  the lens from the technical paper (Jacobian of future-token logits w.r.t.
  residual state, averaged over a fitting corpus); budget +2 weeks.
- Limited GPU memory means 3B requires BF16 and 7B is not yet validated →
  precision pilots and fixed-batch manifests are required for every size.
- Apparent collisions may be lens noise → E9 controls run alongside every
  phase; a collision claim requires beating its shuffled-lens baseline.
- Safety-relevant prompts (E10) must stay benign → all latent-state elicitation
  uses controlled, simulated scenarios; no harmful-content generation.
