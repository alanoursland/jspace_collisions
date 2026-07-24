# Lab Report: GPU Model-Size Series

Date: 2026-07-23  
Models: Qwen2.5 base 1.5B and 3B  
Hardware: RTX 3080 Ti, 12 GiB VRAM; 96 GiB CPU RAM  
Result classification: heterogeneous size effect; one causal near-collision

## Question

Does J-space global faithfulness change with model size within a matched
pretrained model family?

## Fitting configuration

| Model | Precision | Source layers | Prompts | Sequence length | `dim_batch` | Peak VRAM | Fit time |
|---|---|---|---:|---:|---:|---:|---:|
| 1.5B | FP32 | 4, 8, 12, 16, 20, 24, 26 | 100 | 96 | 8 | 8.239 GiB | 2,199 s cumulative |
| 1.5B | BF16 | 4, 8, 12, 16, 20, 24, 26 | 100 | 96 | 8 | 4.504 GiB | 1,148 s |
| 3B | BF16 | 5, 10, 15, 20, 25, 30, 34 | 100 | 96 | 8 | 8.408 GiB | 2,776 s cumulative |

The normalized layer grids compare similar relative depths. The matched 1.5B
BF16 lens differs from its FP32 lens by 0.94% at L4, 0.40% at L20, and about
0.3% near the top, so lens-fitting precision is not driving the main result.

## Lens-quality results

100-prompt results:

| Model/precision | Evaluation | J-lens pass@1 | J-lens pass@10 | Logit pass@1 | Logit pass@10 |
|---|---|---:|---:|---:|---:|
| 1.5B BF16 | Multihop | 0.17 | 0.55 | 0.16 | 0.39 |
| 1.5B BF16 | Typo | 0.03 | 0.12 | 0.05 | 0.32 |
| 3B BF16 | Multihop | 0.11 | 0.51 | 0.06 | 0.30 |
| 3B BF16 | Typo | 0.04 | 0.15 | 0.05 | 0.35 |

Both sizes show a broad-rank multihop advantage and no typo advantage.
Association remains essentially null.

## Behavioral competence gate

Both-correct rates under matched BF16 inference:

| Category | 1.5B | 3B |
|---|---:|---:|
| Role reversal | 47% | 3% |
| Relation binding | 42% | 15% |
| Negation | 42% | 17% |
| Causal flip | 10% | 20% |
| Polysemy | 32% | 37% |
| Safety latent | 33% | 11% |

The larger base model is not uniformly more competent on this answer format.
Collision-rate comparisons must therefore be conditioned on competence.

## 1.5B result

At L20, `role_130` remains the closest causally active candidate under matched
BF16:

| Metric | J-lens | Random transport | Logit lens |
|---|---:|---:|---:|
| Final JS | 0.018 | 0.096 | 0.015 |
| All-position scan JS | 0.042 | 0.323 | 0.160 |
| Bag JS | 0.016 | 0.099 | 0.058 |

Behavior JS is 0.541. Patch transfer:

| Patch condition | Mean transfer |
|---|---:|
| Final statement token | 0.00 |
| Full statement | 0.99 |
| 90%-mass visible subspace | 0.83 |
| Complement | 0.39 |

The pair is substantially closer under the fitted lens than under the
position-aware controls and is causally active. It remains a **near-collision**,
not a strict scan-JS `< 0.02` collision. The visible/complement decomposition
is sensitive to the singular-mass cutoff.

## 3B result

At the preselected L25 layer:

- 25 unique pairs produced both-correct answer flips;
- zero passed the strict all-position collision threshold;
- the closest candidate was `bind_owed_023`, with scan JS 0.138 and behavior
  JS 0.321.

An exploratory sweep over all seven fitted layers produced 175 eligible
pair-layer records from those 25 unique pairs and still found zero strict
all-position collisions. Several final-token distances were tiny, but their
scan distances were approximately 0.6-0.7.

## Interpretation

The observed size effect is not monotonic:

- 1.5B contains a stable, causally active, control-separated near-collision.
- 3B has no comparably tight candidate in the completed search.
- 3B is also much less competent on the role/binding probes, limiting the
  number and strength of meaningful behavioral contrasts.

The data do not justify the claim that larger models are globally more
faithful. They justify the narrower claim that the failure geometry and probe
competence change with size.

## Next tests

1. Run the 3B-Instruct checkpoint to test whether poor probe competence is a
   base-model answer-format issue.
2. Expand paraphrases and seeds around `role_130`.
3. Plot continuous behavior distance against scan distance and controls rather
   than relying on one binary threshold.
4. Test additional normalized layers or models only after passing the
   competence gate.

## Evidence

- [`docs/notebook/2026-07-23-provenance-and-gpu-handoff.md`](../docs/notebook/2026-07-23-provenance-and-gpu-handoff.md)
- [`docs/gpu_campaign.md`](../docs/gpu_campaign.md)
- [`data/lens/qwen2.5-1.5b_wikitext100_grid7_bf16_db8.manifest.json`](../data/lens/qwen2.5-1.5b_wikitext100_grid7_bf16_db8.manifest.json)
- [`data/lens/qwen2.5-3b_wikitext100_grid7_bf16_db8.manifest.json`](../data/lens/qwen2.5-3b_wikitext100_grid7_bf16_db8.manifest.json)

