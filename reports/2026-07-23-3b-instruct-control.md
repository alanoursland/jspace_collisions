# 3B-Instruct Control

Date: 2026-07-23  
Experiment family: E11 (cross-model and training-stage comparison)

## Question

The Qwen2.5-3B base checkpoint was unexpectedly weak on several synthetic
behavioral probes. This control asks whether its negative collision result was
mainly an artifact of answer-format competence, rather than a property of
J-space geometry.

## Design

The control uses `Qwen/Qwen2.5-3B-Instruct` at immutable revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. It has the same 3.086B-parameter
architecture as the 3B base checkpoint.

Two presentation conditions separate training stage from prompt formatting:

- **raw**: the exact prompt strings and direct continuation scoring used for
  the base checkpoint;
- **chat**: the tokenizer-native user/assistant template.

The primary J-lens was fitted on 100 raw WikiText prompts in BF16 at layers
`[5, 10, 15, 20, 25, 30, 34]`, sequence length 96, and `dim_batch=8`. The raw
condition is the matched checkpoint control. Chat evaluation with this
raw-context lens is secondary and tests transfer across prompt contexts.

## Behavioral competence

Percentages are the fraction of pairs for which both answers were correct.

| category | 3B base, raw | 3B-Instruct, raw | 3B-Instruct, chat |
|---|---:|---:|---:|
| causal flip | 20% | 60% | 30% |
| negation | 17% | 67% | 83% |
| polysemy | 37% | 42% | 74% |
| relation binding | 15% | 54% | 19% |
| role reversal | 3% | 59% | 15% |
| safety latent | 11% | 44% | 0% |

Instruction tuning substantially restores competence under the original raw
protocol, especially for role reversal and relation binding. Native chat
formatting is not a universal correction: it helps negation and polysemy but
hurts role reversal, binding, and the current safety-latent probe.

## Lens fit and convergence

- 100-prompt cumulative fit time: 2,546 seconds
- peak CUDA allocation: 8.408 GiB
- lens SHA-256:
  `01b9abb101d1f459bf5349b0e1dcc03a4381b4444697a0f6108901d2d046c59b`
- 20-to-100-prompt relative Frobenius movement by layer:
  `.380, .352, .322, .335, .288, .194, .044`
- late-prompt mean changes: approximately 0.98-1.61%

Early-layer transports remain less converged than late-layer transports, as in
the base-model size series. The immutable fit settings and checksum are in the
[100-prompt manifest](../data/lens/qwen2.5-3b-instruct_wikitext100_grid7_bf16_db8.manifest.json).

## E1 readout results

| condition/eval | J-lens pass@1 | J-lens pass@10 | logit pass@1 | logit pass@10 |
|---|---:|---:|---:|---:|
| raw multihop | .08 | .47 | .06 | .29 |
| raw typo | .03 | .11 | .07 | .38 |
| raw association | .00 | .00 | .00 | .00 |
| chat multihop | .09 | .32 | .04 | .22 |
| chat typo | .00 | .03 | .00 | .03 |
| chat association | .00 | .00 | .00 | .00 |

The raw J-lens retains a broad-rank multihop advantage over the logit lens but
does not show a general readout advantage. Chat scores are reported only as
transfer results because the lens fitting corpus used raw contexts.

## E2 collision search

The preselected comparison layer was L25. A strict collision requires a
both-correct answer flip, high behavior distance, and low maximum aligned
position J-distance. Legacy counts use only the final statement token.

| condition | competent answer flips | legacy final-token hits | strict all-position hits |
|---|---:|---:|---:|
| raw | 160 | 27 | 0 |
| chat | 48 | 22 | 0 |

The nearest eligible raw candidate was `role_129`:

- final-token JS: .238
- all-position scan JS: .301
- behavior JS: .610
- random-transport scan JS: .367
- logit-lens scan JS: .341

An interim 20-prompt candidate, `role_011`, reached scan JS .104, but it did
not persist after the fit was extended to 100 prompts. The converged raw
control therefore has no candidate comparable to the 1.5B `role_130`
near-collision (scan JS .042 in matched BF16).

The closest chat record, `bind_owed_023`, illustrates the inert-site failure
mode:

- final-token JS: .007
- all-position scan JS: .612
- behavior JS: .693
- random-transport scan JS: .400
- logit-lens scan JS: .262

Chat produced many apparently strong final-token or bag-of-positions
collisions, but all failed the all-position scan. The nearest eligible record
was also farther apart under the J-lens than under both applicable controls.

## Interpretation

The base model's weak behavioral competence was partly a training-stage and
answer-following effect: the matched Instruct checkpoint is much stronger
under the original raw protocol. Correcting that confound does **not** reveal
a stable, tight 3B all-position collision. The 3B negative result therefore
survives this control within the tested prompt bank and preselected layer.

The chat condition strengthens a separate methodological conclusion:
final-token similarity can be abundant while the complete aligned statement
contains a large distinguishing signal. Collision claims should use
position-aware distances and applicable controls, not final-token counts.

## Limitations and next tests

- E2 used the preselected L25 layer rather than a newly selected layer on the
  Instruct checkpoint.
- The chat condition reuses a raw-context lens and is not evidence about a
  separately fitted chat-context Jacobian lens.
- The prompt bank is synthetic and competence varies strongly by category.
- No causal patch was run because neither condition produced an applicable
  all-position near-collision.
- A stronger test would expand paraphrases/seeds around the 1.5B candidate and
  report continuous scan-distance versus behavior-distance curves.

See the [methodology audit](2026-07-23-methodology-and-control-audit.md) for
the collision criteria and control rationale.
