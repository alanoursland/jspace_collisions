# Replication and Instrumentation

Program family: E1 (+E9 controls) | Plan: Phase 1 | Status: 0.5B baseline complete; GPU scaling in progress

Establish a reliable baseline: integrate the reference J-lens implementation,
reproduce the paper's qualitative examples (multi-hop latent concepts, silent
arithmetic, bug detection, injection recognition, concept swaps) on an open
Qwen-family model, and build the activation recording pipeline.

Exit criteria: paper examples reproduced; readout metrics stable across seeds
and paraphrases; first activation dataset with manifest in `data/`.

Current result: on Qwen2.5-0.5B, multihop pass@1 is 0.34 and typo pass@1
is 0.16 with the final 100-prompt lens, versus approximately zero for the
applicable shuffled controls. See the dated entries in `docs/notebook/`.

See `docs/research_program.md` for the full procedure.
