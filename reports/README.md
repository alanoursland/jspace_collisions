# Lab Reports

This directory contains concise, result-oriented reports derived from the
dated lab notebook and generated experiment summaries. Reports distinguish:

- **observations** directly measured in an experiment;
- **interpretations** supported by those observations;
- **limitations** that constrain the claim;
- **next tests** that could strengthen or falsify the interpretation.

The detailed chronological record remains in [`docs/notebook/`](../docs/notebook/).
Generated payloads under `results/` are intentionally not committed; fitted
lens manifests record immutable model revisions, fit settings, and artifact
checksums.

## Reports

| Date | Report | Main result |
|---|---|---|
| 2026-07-22 | [0.5B CPU baseline](2026-07-22-0.5b-cpu-baseline.md) | Original final-token collision was attached to a causally inert site |
| 2026-07-23 | [GPU model-size series](2026-07-23-gpu-model-size-series.md) | 1.5B has a causal near-collision; 3B has no comparably tight candidate |
| 2026-07-23 | [Methodology and control audit](2026-07-23-methodology-and-control-audit.md) | Position-aware monitoring and applicable controls materially revise the claims |

## Current overall conclusion

The experiments do not support treating the Jacobian-lens readout as a
globally faithful coordinate system. They also do not yet establish a broad
class of hidden J-space fibers. The strongest result is narrower: at 1.5B,
one behaviorally active prompt pair remains unusually close under the
position-aware J-lens relative to controls, but its classification is
threshold-sensitive. At 0.5B and 3B, no strict all-position collision was
found in the completed sweeps.

