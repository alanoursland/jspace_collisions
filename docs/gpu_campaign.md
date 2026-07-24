# GPU Campaign: Model-Scale Tests

Status: first 0.5B/1.5B/3B size-series pass complete (2026-07-23).

## Hardware

- NVIDIA GeForce RTX 3080 Ti, 12 GiB VRAM
- 96 GiB CPU RAM
- PyTorch 2.11.0 + CUDA 12.8
- Model weights and durable Jacobian accumulators may live in CPU RAM, but
  the retained forward/backward graph must fit in VRAM.

## Scientific priority

The first cross-model axis is model size within one pretrained family:

1. `Qwen/Qwen2.5-0.5B` (existing 100-prompt lens and baseline results)
2. `Qwen/Qwen2.5-1.5B`
3. `Qwen/Qwen2.5-3B`

Holding model family and training stage fixed makes size the primary changed
variable. Base-vs-instruct and cross-family comparisons follow only after the
size series is working.

The official Qwen2.5 family provides matched base and instruction-tuned
checkpoints at 0.5B, 1.5B, 3B, and larger sizes. A 7B fit is not an initial
target: BF16 weights alone exceed the local 12 GiB VRAM budget, and ordinary
inference-style CPU offload is not assumed to support this retained-gradient
estimator correctly.

## Stage 0: runtime validation

- [x] Pin the Anthropic reference implementation at commit `581d398`.
- [x] Pass the 22 local unit tests and 32 reference tests.
- [x] Load Qwen2.5-0.5B on CUDA and run an end-to-end E1 smoke item.
- [x] Benchmark 1.5B precision, layer-grid, and `dim_batch` choices.
- [x] Benchmark 3B at `dim_batch` 2, 4, and 8.

Selected production settings:

- 1.5B: FP32 production/convergence series plus a matched BF16 control,
  sequence length 96, `dim_batch=8`, layers `[4, 8, 12, 16, 20, 24, 26]`
- 3B: BF16, sequence length 96, `dim_batch=8`, layers
  `[5, 10, 15, 20, 25, 30, 34]`
- 3B pilot: 28 seconds/prompt and 8.405 GiB peak CUDA allocation
- 3B 100-prompt cumulative fit: 2,776 seconds

BF16 pilot transports varied with `dim_batch` by up to about 4% at early
layers and under 1% near the top. All 3B production artifacts therefore use
one fixed batch setting. The completed 100-prompt 1.5B BF16 control differs
from its FP32 transport by only 0.94% at L4 and 0.40% at L20, falling to about
0.3% near the top.

## Stage 1: matched size-series lenses

For each feasible model:

1. Fit 20 prompts with checkpointing.
2. Snapshot interim lenses at 10 and 20 prompts.
3. Run E1 lens-quality evaluations and compare rank metrics.
4. If the 10-to-20-prompt change is still material, extend to 50, then 100.
5. Record model revision, dtype, device, source layers, sequence length,
   `dim_batch`, fit time, and lens checksum in the manifest.

Use normalized layer locations when comparing models with different depths.
Do not compare raw layer numbers as though they represented the same stage of
computation.

Completed artifacts:

- 1.5B FP32 lenses at 20, 50, and 100 prompts
- 1.5B BF16 precision-control lens at 100 prompts
- 3B BF16 lenses at 20, 50, and 100 prompts
- E1 evaluations for both sizes
- position-aware E2 sweeps at the matched mid/late layers
- an exploratory seven-layer E2 sweep for the 100-prompt 3B lens

## Stage 2: competence gate

Before collision rates are compared, report both-correct rates by prompt
category. A category is excluded from collision claims when the model cannot
reliably perform its behavioral probe.

This gate is especially important because the 0.5B base model had useful
competence on role reversal but essentially none on negation, causal-flip, or
safety-latent probes.

## Stage 3: position-aware collision search

The original E2 sweep measured the final statement token. Patch confirmation
showed that this site was causally inert for the strongest role-reversal
examples: behavior was carried at earlier name positions.

Larger-model collision candidates must therefore pass all of these checks:

1. Low readout distance at the proposed site.
2. Low readout distance under a position-aware or all-position monitor.
3. High behavioral distance with both variants answered correctly.
4. A full activation swap at the proposed site transfers behavior.
5. The candidate separates from the applicable random-transport/logit-lens
   distance controls.

A final-token-only collision may still be reported as a monitoring caveat, but
not as evidence for a behaviorally active J-space fiber.

## Stage 4: causal decomposition

For patch-confirmed candidates:

- repeat full, lens-visible, and low-gain/fiber swaps;
- sweep singular-mass thresholds rather than fixing 90%;
- report per-position readout distances;
- test paraphrases and prompt-bank seeds;
- compare matched normalized layers across model sizes.

## Decision gates

- Reduce `dim_batch` after CUDA OOM; do not change sequence length and
  `dim_batch` simultaneously when estimating scaling.
- Stop a full fit if the pilot projects to an impractical runtime; use fewer
  prompts and demonstrate convergence instead of silently weakening the
  corpus.
- Do not use quantized weights for lens fitting unless a separate validation
  shows that the fitted readout agrees with an unquantized checkpoint.
- Do not treat 7B CPU-offloaded inference as evidence that 7B Jacobian fitting
  is supported; retained-gradient execution must be validated separately.
