# data/

Datasets and activation caches. Payloads are gitignored; only READMEs and
`manifest.json` files are tracked.

Planned layout (Phase 1):

```
data/
  activations/<run_id>/
    manifest.json      # model, revision, prompts, layers, positions, dtype, seed
    *.safetensors      # residual activations, lens logits (gitignored)
  benchmarks/
    collision_pairs_v1.jsonl   # frozen export of jspace.prompts banks
```

Rule: anything needed to regenerate a payload (model id + revision, prompt
set, seed, code version) must be in its manifest.
