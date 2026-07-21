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

This development container is CPU-only. It supports: all pure-Python library
work, prompt benchmark construction, metric development, and smoke tests with
sub-1B open models (e.g. Qwen2.5-0.5B) if network policy permits HuggingFace
downloads. Lens fitting and the main experiment battery need a GPU box; the
repo is structured so experiments are launchable scripts that run identically
there.

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
- [ ] Obtain and pin the Anthropic reference implementation
      (github.com/anthropics/jacobian-lens). Not yet accessible from this
      session (GitHub scope is limited to this repo); either add it as a
      session source, vendor a pinned copy, or add it as a git dependency.

## Phase 1 — Baseline / replication (program Month 1; E1, E9 controls)

Goal: a working J-lens on at least one open model, with trustworthy metrics.

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

1. Resolve access to `anthropics/jacobian-lens` and pin a version; write the
   `JLens` adapter against the real API (interfaces in `lens/interface.py` are
   currently written from the paper's description and must be reconciled).
2. Pick the primary small model (default: smallest Qwen model the reference
   implementation supports) and verify weights download in the target compute
   environment.
3. Grow prompt banks from starter size (~10/category) toward 100/category,
   each pair with a behavioral probe and scoring rule.
4. Decide GPU environment for Phase 1 lens fitting.

## Risk register

- Reference implementation API may not match our interfaces → adapters are
  thin by design; reconcile in Phase 1 before any experiment code depends on
  details.
- Reference implementation may not be public/accessible → fallback: implement
  the lens from the technical paper (Jacobian of future-token logits w.r.t.
  residual state, averaged over a fitting corpus); budget +2 weeks.
- No GPU in this container → experiments are scripts, not notebooks; run them
  on a GPU box unchanged. Smoke-test path on CPU with a 0.5B model.
- Apparent collisions may be lens noise → E9 controls run alongside every
  phase; a collision claim requires beating its shuffled-lens baseline.
- Safety-relevant prompts (E10) must stay benign → all latent-state elicitation
  uses controlled, simulated scenarios; no harmful-content generation.
