# J-Lens Method (extracted from the technical paper)

Source: https://transformer-circuits.pub/2026/workspace/index.html
(fetched 2026-07-21; this summary is the reference for our reimplementation
until the official code in `anthropics/jacobian-lens` can be reconciled).

## Definition

For layer ℓ, the lens matrix is the corpus- and position-averaged Jacobian of
the final-layer residual stream with respect to the layer-ℓ residual stream:

    J_ℓ = E_{prompt, t, t' >= t} [ ∂h_{final, t'} / ∂h_{ℓ, t} ]

- Expectation over: prompts from a pretraining-like corpus (paper: 1,000
  prompts), source positions t, and all subsequent positions t' >= t.
- Averaging over contexts isolates verbalizable *dispositions* (what an
  activation tends to promote saying later) from context-specific uses.

## Readout

    lens(h_ℓ) = softmax( W_U · norm( J_ℓ · h_ℓ ) )

where W_U is the unembedding and norm is the model's final layer norm.
The J-lens *vectors* (per-token directions) are the rows of W_U J_ℓ.

## J-space contents

The paper defines active J-space contents via sparse nonnegative
decomposition of J_ℓ h onto lens vectors using gradient pursuit, typically
k <= 25 active vectors.

## Paper hyperparameters

- Corpus: 1,000 pretraining-like prompts
- Layers: 25 evenly spaced, reindexed to [0, 100]
- Sparse decomposition: k <= 25, gradient pursuit
- Models: Claude Sonnet 4.5 (primary), Haiku 4.5, Opus variants

## Reference implementation (now in use — supersedes the plan above to
## approximate with random probes)

We use the reference code directly via the fork `alanoursland/jacobian-lens`
(commit 581d398, installed editable; cloned at /workspace/jacobian-lens).
Its exact estimator (`jlens/fitting.py`):

- One forward per prompt with the sequence replicated `dim_batch` times;
  `ceil(d_model / dim_batch)` backward passes compute `dim_batch` rows of
  J_l each, for all source layers simultaneously.
- Cotangents are one-hot in an output dimension, set at *every valid target
  position at once*: the gradient at source position p is
  `sum_{p' >= p} dh_final[p'] / dh_l[p]`, then averaged over p. (Sum over
  targets, mean over sources — the paper's reduction.)
- The first 16 positions (attention sinks) and the final position are
  excluded from the average.
- Readout: `unembed(J_l @ h)` where `unembed = lm_head(final_norm(·))` —
  the norm placement matches the paper formula.

Our deviations from the paper's setting (CPU budget, not method):
- open 0.5B-scale model instead of Claude-scale
- corpus ~100 wikitext prompts at max_seq_len 96 (paper: 1000 x 128;
  README notes quality saturates quickly and ~100 prompts is usable)
