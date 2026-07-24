# 2026-07-23 — Project provenance and GPU handoff

## Provenance

The research program began with a deliberately speculative analogy. The user
asked whether Anthropic's J-space was connected to the Jacobian Conjecture.
The answer was that the naming/technical connection was coincidental, but the
local-to-global distinction could still motivate a useful stress test:

> Local sensitivity is not global faithfulness.

The resulting program was written to `docs/research_program.md`. While the
user was on vacation, Claude Code remote servers implemented the CPU-feasible
part of that program. The remote environment had no GPU, leading to the
Qwen2.5-0.5B experiments and the long checkpointed CPU lens fit in the first
research sprint.

During the same week, Levent Alpöge publicly announced an explicit
three-dimensional counterexample to the Jacobian Conjecture, credited to
Claude Fable. The finite certificate has since been independently checked and
formalized. This mathematical result does not imply anything about J-space,
but it makes the project's motivating local-versus-global analogy unusually
literal.

References:

- https://jacobianfun.org/jacobian-explained
- https://arxiv.org/abs/2607.18186

## Local GPU handoff

Hardware discovered:

- RTX 3080 Ti with 12 GiB VRAM
- 96 GiB CPU RAM

The local Anthropic checkout is `E:\Projects\jacobian-lens`, clean and exactly
at the manifest-pinned commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

The experiment loaders were updated to accept device and dtype selection.
Activation captures are transferred to FP32 CPU tensors after each forward so
the existing NumPy lens/metric boundary remains valid on CUDA. The E1 runner's
remote-only `/workspace/jacobian-lens` path was replaced with discovery from
the installed `jlens` package.

Validation:

- local tests: 19 passed
- Anthropic reference tests: 32 passed
- CUDA runtime: PyTorch 2.11.0, CUDA 12.8
- Qwen2.5-0.5B E1 smoke: completed on GPU

Next: benchmark a one-prompt Qwen2.5-1.5B fit, then use the measured memory and
runtime to set the full matched-size campaign in `docs/gpu_campaign.md`.

## Position-aware regression check

Before scaling E2, the sweep was extended with aligned all-position metrics.
The strongest old example gives:

| pair/layer | final JS | scan JS (max position) | bag JS | min position top-20 overlap | behavior JS |
|---|---:|---:|---:|---:|---:|
| role_025 / L16 | 0.00275 | 0.64783 | 0.15215 | 0.02564 | 0.61401 |

The model answers both variants correctly and flips its answer. The legacy
final-token metric says the readouts are almost identical, but an all-position
monitor finds an almost maximally separating site. This independently
quantifies the patch-confirmation conclusion: the clean final-token collision
is real but attached to the wrong causal site.

Full L16 regression sweep (311 pairs, final 100-prompt lens, BF16 CUDA model):

- 121 pairs produced both-correct answer flips; 119 had equal token counts and
  therefore supported the aligned-position metric.
- The legacy threshold (final JS < 0.02, behavior JS > 0.5) selected 5
  collisions.
- The all-position threshold (scan JS < 0.02 with the same behavior gate)
  selected 0 collisions.
- The five legacy candidates had scan JS from 0.334 to 0.648.
- Among aligned, both-correct flips, the smallest observed scan JS was 0.143.

At this model/layer/threshold, the entire headline collision class disappears
when the monitor is allowed to inspect every aligned statement position.

## First 1.5B lens

A seven-layer Qwen2.5-1.5B lens was fitted on 20 WikiText prompts at normalized
workspace layers `[4, 8, 12, 16, 20, 24, 26]`.

Fit settings and performance:

- FP32 model and Jacobian computation
- `dim_batch=8`
- sequence length 96
- 437 seconds total (about 22 seconds per prompt)
- peak CUDA allocation 8.239 GiB

E1 results:

| eval | lens | pass@1 | pass@10 |
|---|---|---:|---:|
| multihop | J-lens | 0.17 | 0.54 |
| multihop | logit lens | 0.16 | 0.39 |
| multihop | random transport | 0.00 | 0.03 |
| typo | J-lens | 0.03 | 0.17 |
| typo | logit lens | 0.05 | 0.31 |
| association | J-lens | 0.00 | 0.01 |

The lens adds multihop information above the logit lens at broader ranks, but
does not improve pass@1 and underperforms the logit lens on typo. Late-prompt
running-mean shifts remained around 0.06-0.09, so the lens is being extended
to 50 prompts before this is interpreted as a model-size effect.

At 50 prompts, multihop pass@1 remained 0.17 and pass@10 rose slightly from
0.54 to 0.56. Typo pass@10 fell from 0.17 to 0.15, and association remained
null. Thus the E1 outcome is stable enough to use the 50-prompt lens for an
initial E2 sweep.

The underlying transports are less converged than the readout scores:
20-to-50-prompt relative Frobenius changes are 0.30, 0.28, 0.26, 0.23, 0.21,
0.11, and 0.04 from L4 through L26. Any fine-grained geometric claim at early
layers still needs the planned 100-prompt lens.

## 1.5B position-aware collision and patch results

At L20 with the 50-prompt lens:

- role-reversal both-correct competence: 87% (0.5B L16: 60%)
- relation-binding both-correct competence: 66% (0.5B: 25%)
- negation both-correct competence: 42% (0.5B: 0%)
- legacy final-token collisions at JS < 0.02 and behavior JS > 0.5: 6
- all-position collisions at scan JS < 0.02 with the same gate: 0

The closest causal candidate is `role_130`:

- prompts: "David trusted Frank." / "Frank trusted David."
- final JS 0.011
- all-position scan JS 0.030
- behavior JS 0.669
- random-transport scan JS 0.322
- logit-lens scan JS 0.169
- full-statement patch transfer 1.00
- final-token patch transfer 0.00

This is a causal near-collision under distributional JS distance and is
substantially closer under the fitted J-lens than under its distance controls.
It is not invisible under every metric: minimum per-position top-20 overlap is
0.379.

The visible/fiber decomposition is strongly threshold-dependent:

| singular mass called visible | visible rank | visible transfer | complement transfer |
|---:|---:|---:|---:|
| 50% | 127 | 0.01 | 1.00 |
| 75% | 360 | 0.11 | 0.99 |
| 90% | 664 | 0.86 | 0.34 |
| 99% | 1171 | 0.99 | 0.02 |

The behaviorally decisive difference lies largely in middle singular modes
between the 75% and 90% cumulative-mass cutoffs. A binary "visible versus
fiber" claim at one arbitrary threshold is not robust; follow-up should report
a transfer-versus-singular-rank curve.

## 100-prompt convergence

The 1.5B lens was extended to 100 prompts. By prompts 97-100, the maximum
per-prompt running-mean change was approximately 0.96-1.13%.

E1 remained stable:

- multihop J-lens pass@1/pass@10: 0.17/0.55
- multihop logit-lens pass@1/pass@10: 0.16/0.39
- typo J-lens pass@10: 0.12 (logit lens: 0.31)
- association remained null

The 50-to-100-prompt transport movement was still 0.16 at L4 and 0.10 at L20,
falling to 0.02 at L26. Readout rankings converge earlier than the raw
transport matrices.

The L20 sweep and `role_130` causal result also survived qualitatively:

- scan JS changed from 0.030 to 0.039
- final JS changed from 0.011 to 0.021
- full-statement transfer remained 1.00
- 90%-mass visible/complement transfer changed only from 0.86/0.34 to
  0.88/0.31

The pair is stable as a ranked causal near-collision, but not as a binary
member of an arbitrary final-JS < 0.02 class. Continuous distances and control
comparisons are the defensible report.
