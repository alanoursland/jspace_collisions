# jspace_collisions

Testing the global faithfulness of J-space: is the Jacobian-lens readout a
faithful global coordinate system for a language model's verbalizable
workspace, or a locally useful projection with hidden collisions, folds, and
blind spots?

The guiding question:

> If two internal states have the same or nearly the same J-lens readout, are
> they equivalent for reporting, reasoning, and safety-relevant behavior?

Full motivation, hypotheses (H1–H5), and the twelve experiment families are in
[`docs/research_program.md`](docs/research_program.md). The phased execution
plan is in [`PLAN.md`](PLAN.md).

## Repository layout

```
docs/                  Research program, metric definitions, lab notebook
src/jspace/            Shared library
  metrics/             J-distance and behavior-distance metrics
  prompts/             Minimal-pair prompt generators (collision benchmark)
  lens/                J-lens interfaces and adapters
  patching/            Activation patching harness
experiments/           One directory per experiment family (e01–e12)
tests/                 Unit tests for the shared library
scripts/               Environment checks and utilities
data/                  Datasets and activation caches (gitignored payloads)
results/               Experiment outputs (gitignored payloads)
```

## Setup

Requires Python 3.11+. Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"        # core: numpy; dev: pytest, ruff
uv pip install -e ".[models]"     # torch + transformers, for model work
pytest
python scripts/env_check.py
```

The core library (metrics, prompt generation) is pure NumPy and runs anywhere.
Model-dependent code (`lens/`, `patching/`) requires the `[models]` extra and,
for anything beyond ~1B-parameter smoke tests, a GPU.

## External references

- Anthropic summary: https://www.anthropic.com/research/global-workspace
- Technical paper: https://transformer-circuits.pub/2026/workspace/index.html
- Reference implementation: https://github.com/anthropics/jacobian-lens

The reference implementation is pinned at commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`; this project imports it as the
separate editable `jlens` package rather than vendoring it.

## Status

The 0.5B CPU phase is complete: the core readout result replicated, apparent
final-token collisions were found, and causal patching showed that the
strongest collision site was behaviorally inert. GPU model-scaling work is now
in progress; see `docs/gpu_campaign.md` and `PLAN.md`.
