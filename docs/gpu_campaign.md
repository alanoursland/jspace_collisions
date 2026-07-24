# GPU Campaign: Model-Scale Tests

Status: infrastructure validation in progress (2026-07-23).

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
- [x] Pass the 19 local unit tests and 32 reference tests.
- [x] Load Qwen2.5-0.5B on CUDA and run an end-to-end E1 smoke item.
- [ ] Benchmark one 1.5B prompt at four representative source layers.
- [ ] Benchmark one 3B prompt only if the 1.5B memory/timing result is sound.

Pilot settings:

- BF16 model weights and activations
- sequence length 48
- `dim_batch=2` initially
- source layers spread through the workspace band
- one fitting prompt

The pilot measures peak VRAM and time per prompt. It is not a research lens.

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

