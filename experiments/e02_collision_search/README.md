# J-Space Collision Search

Program family: E2 | Plan: Phase 2 | Status: not started

Find prompt/activation pairs with near-identical J-lens readouts but
different behavior. Uses `jspace.prompts` minimal pairs, `jspace.metrics`
J-distance and behavior-distance, ranked by `collision_score`. Top candidates
are confirmed by cross-patching activations within each pair.

Primary outcome: collision rate by task class, layer, position. This family
is the core of the first concrete study ("Hidden Folds in J-Space").

See `docs/research_program.md` for the full procedure.
