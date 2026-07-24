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

- local tests: 22 passed
- Anthropic reference tests: 32 passed
- CUDA runtime: PyTorch 2.11.0, CUDA 12.8
- Qwen2.5-0.5B E1 smoke: completed on GPU

The subsequent 1.5B and 3B campaigns below used these validated loaders.

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

## 3B production lens

`Qwen/Qwen2.5-3B` revision
`3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` was fitted at normalized layers
`[5, 10, 15, 20, 25, 30, 34]`, sequence length 96, in BF16.

One-prompt `dim_batch` pilots:

| dim_batch | seconds | peak CUDA GiB |
|---:|---:|---:|
| 2 | 77 | 6.550 |
| 4 | 40 | 7.209 |
| 8 | 28 | 8.405 |

Batch 2 versus 4 and batch 4 versus 8 differed by approximately 3-4% in
relative Frobenius norm at the earliest layer, falling below 1% near the top.
Production fits therefore fixed `dim_batch=8`; BF16 batch sensitivity remains
a recorded numerical uncertainty.

The 100-prompt lens took 2,776 cumulative fit seconds across resumable
1→20→50→100 runs. The final peak allocation was 8.408 GiB. Late-prompt
running-mean changes were approximately 1.1-1.5%, with occasional larger
prompt outliers.

Transport convergence:

| comparison | L5 | L10 | L15 | L20 | L25 | L30 | L34 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20→50 | .318 | .288 | .265 | .279 | .241 | .161 | .035 |
| 50→100 | .220 | .205 | .188 | .194 | .137 | .083 | .018 |

As at 1.5B, raw early-layer transports converge more slowly than downstream
readout metrics.

## 3B E1 and competence

The 100-prompt E1 results were stable relative to the 50-prompt lens:

| eval | J-lens pass@1 | J-lens pass@10 | logit pass@1 | logit pass@10 |
|---|---:|---:|---:|---:|
| multihop | .11 | .51 | .06 | .30 |
| typo | .04 | .15 | .05 | .35 |
| association | .00 | .01 | .00 | .02 |

The J-lens gives a broad-rank multihop advantage but not a general readout
improvement. Its multihop pass@1 is lower than the 1.5B result (.17), while
its pass@10 is similar (.51 versus .55).

The 3B base model was unexpectedly weak on the synthetic behavioral probes:

| category | 3B BF16 | 1.5B BF16 | 1.5B FP32 |
|---|---:|---:|---:|
| role reversal | 3% | 47% | 87% |
| relation binding | 15% | 42% | 66% |
| negation | 17% | 42% | 42% |
| causal flip | 20% | 10% | 10% |
| polysemy | 37% | 32% | 32% |
| safety latent | 11% | 33% | 33% |

This is non-monotonic benchmark competence, not evidence that scale itself
damages binding. The base model may not follow the probe answer format
consistently; an instruct-model axis is now a high-priority control.

## 3B position-aware collision result

At the preselected L25 comparison layer, the 100-prompt sweep produced 25
unique both-correct answer-flip pairs but no legacy or strict collision under
the existing behavior/distance thresholds. The closest all-position record
was `bind_owed_023`:

- final JS 0.032
- scan JS 0.138
- bag JS 0.059
- minimum position top-20 overlap 0.176
- behavior JS 0.321

An exploratory sweep across all seven fitted layers produced 175 eligible
pair-layer records from the same 25 unique pairs, again with zero strict
all-position collisions. Several early-layer final-token distances were very
small, but their all-position scan distances were approximately 0.6-0.7.
This reproduces the inert-site failure mode of the original 0.5B collision
claim rather than finding a new globally hidden state.

The defensible first size-series conclusion is therefore heterogeneous:

- 0.5B: clean final-token artifacts, no strict all-position collision.
- 1.5B: a stable, causally active near-collision (`role_130`, scan JS
  .030→.039) that beats readout controls but is threshold-sensitive.
- 3B: no comparably tight all-position candidate in the 100-prompt,
  seven-layer scan; closest scan JS .138, with a competence-limited probe set.

## Matched 1.5B BF16 control

A 100-prompt 1.5B lens was refitted under the 3B numerical regime: BF16,
`dim_batch=8`, sequence length 96, and the same normalized seven-layer grid.
It took 1,148 seconds and peaked at 4.504 GiB.

Relative Frobenius differences from the 1.5B FP32 lens were:

| L4 | L8 | L12 | L16 | L20 | L24 | L26 |
|---:|---:|---:|---:|---:|---:|---:|
| .0094 | .0071 | .0063 | .0058 | .0040 | .0029 | .0032 |

The E1 scores were effectively identical to FP32. BF16 model inference did
reduce confidence/competence on some behavioral probes, but the 1.5B-versus-3B
separation remained large (for example, role-reversal both-correct 47% versus
3%).

`role_130` also survived the matched regime:

- final JS 0.018
- scan JS 0.042
- bag JS 0.016
- behavior JS 0.541
- random-transport/logit-lens scan JS 0.323/0.160
- final-token/full-statement patch transfer 0.00/0.99
- 90%-mass visible/complement transfer 0.83/0.39

Thus the 1.5B near-collision is not a FP32 artifact. The remaining limitation
on the size comparison is the 3B base model's weak competence on this prompt
format, not lens-fitting precision.

## Matched 3B-Instruct control

`Qwen/Qwen2.5-3B-Instruct` revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1` was evaluated under both the
original raw prompts and the tokenizer-native chat template. Raw prompting
substantially restored behavioral competence relative to the 3B base model:
both-correct role-reversal accuracy rose from 3% to 59%, relation binding from
15% to 54%, and negation from 17% to 67%. Chat formatting was mixed rather
than uniformly better.

A matched 100-prompt raw-context BF16 lens was fitted at layers
`[5, 10, 15, 20, 25, 30, 34]`. It took 2,546 cumulative seconds and peaked at
8.408 GiB. At preselected L25, the raw E2 sweep produced 160 competent
answer-flip pairs and 27 legacy final-token hits, but zero strict all-position
hits. Its nearest eligible candidate had scan JS .301 and behavior JS .610.

The secondary chat transfer condition produced 22 legacy hits but also zero
strict hits. Its closest final-token example had JS .007 yet scan JS .612,
again exposing a strong distinguishing signal elsewhere in the statement.
Because no applicable all-position near-collision survived, no causal patch
was run. The full tables and limitations are in
`reports/2026-07-23-3b-instruct-control.md`.

## Null-control correction

The completed sweep exposed a control-design invariant: applying one fixed
vocabulary permutation to both readouts leaves pairwise JS, cosine, top-k
overlap, and rank overlap unchanged. `ShuffledLens` is therefore valid for E1
token-identity evaluations but vacuous for E2 distance calibration. Future E2
runs exclude it; historical records retain it for provenance, and the
analyzer omits it from distance-control tables. Random transport and the logit
lens remain the applicable E2 controls.
