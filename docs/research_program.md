# Research Program: Testing Global Faithfulness of J-Space

## Purpose

This document proposes a research program to test whether J-space, as revealed
by the Jacobian lens, is a faithful global coordinate system for a language
model's verbalizable workspace or a locally useful projection with hidden
collisions, folds, and blind spots.

The motivating analogy comes from the July 2026 Jacobian conjecture discussion:
even a map with strong local Jacobian structure may fail to be globally
one-to-one. That pure math claim is separate from J-space and does not formally
undermine it. The useful lesson is methodological:

> Local sensitivity is not global faithfulness.

For J-space, the question becomes:

> If two internal states have the same or nearly the same J-lens readout, are
> they equivalent for reporting, reasoning, and safety-relevant behavior?

If yes, J-space is closer to a usable coordinate system for the model's
workspace. If no, J-space is a valuable readout but not a complete state
description.

## Background

Anthropic describes the Jacobian lens, or J-lens, as a method that finds, for
each vocabulary token, the internal activity pattern that makes a model more
likely to say that token later. Applying the lens across model layers yields a
ranked list of latent "words" or concepts that Anthropic calls the contents of
J-space. The Anthropic research post argues that J-space behaves like a global
workspace: information in it can be reported, manipulated, and used by downstream
computation.

Primary sources:

- Anthropic summary: https://www.anthropic.com/research/global-workspace
- Technical paper: https://transformer-circuits.pub/2026/workspace/index.html
- Reference implementation: https://github.com/anthropics/jacobian-lens

The paper also explicitly warns that the current J-lens is incomplete. It is
restricted largely to single-token concepts, treats the workspace as a bag of
independent concept vectors, and may miss richer structure such as binding,
relations, roles, and multi-token concepts.

This plan takes those limitations seriously and turns them into testable
hypotheses.

## Core Research Questions

1. Is the J-lens readout injective enough over behaviorally relevant internal
   states?
2. Do same-J-space states reliably lead to the same reports, decisions, and
   reasoning paths?
3. Do J-space interventions have stable causal effects across contexts, or are
   they context-dependent patches?
4. How large and behaviorally active are the fibers of the J-lens map, meaning
   the sets of hidden states with the same or nearly the same readout?
5. Can safety-relevant internal states be hidden outside J-space or expressed in
   J-space-colliding forms?
6. Can richer lenses, such as template lenses, oracle lenses, relation lenses,
   or SAE-assisted lenses, reduce the collision problem?

## Working Definitions

Let h be a residual-stream activation at a layer and position.

Let L(h) be the J-lens readout, represented as either full lens logits or a
top-k token distribution.

Let B(h) be a behavioral measurement after patching h into a run: final answer,
choice distribution, tool-use decision, refusal behavior, reasoning benchmark
score, or safety classifier result.

Let A(h) be auxiliary internal state: SAE features, attention patterns,
non-J-space residual components, attribution graph features, or template-lens
readouts.

The strongest form of J-space faithfulness would say that behavior approximately
factors through J-space:

    if L(h1) ~= L(h2), then B(h1) ~= B(h2)

The research program searches for counterexamples to this factorization.

## Hypotheses

### H1: J-space is locally causal but globally non-injective

J-lens directions are causally meaningful near ordinary model activations, but
many distinct states share similar J-lens signatures while producing different
downstream behavior.

### H2: Collisions concentrate in binding and relation structure

The single-token bag-of-concepts representation will collide especially often
on prompts with the same entities but different roles:

- "Alice trusts Bob" vs "Bob trusts Alice"
- "France defeated China" vs "China defeated France"
- "the model noticed deception" vs "the model planned deception"

The top tokens may be similar while the relation among them differs.

### H3: Multi-token and abstract concepts are under-represented

Concepts that are diffuse, compositional, or multi-token will either fail to
appear, appear as misleading fragments, or collide with unrelated concepts that
share token parts.

### H4: Safety monitors based only on J-space are bypassable

If monitoring looks only for certain J-space tokens such as "fake",
"manipulation", "deception", or "blackmail", then some functionally similar
states may evade detection by using non-J-space representations, oblique
concepts, or context-dependent encodings.

### H5: Expanded lenses reduce but do not eliminate collisions

Template lenses, oracle lenses, and SAE-assisted readouts should improve
coverage for phrases and abstract concepts, but binding and global structure
will remain partially unresolved.

## Experiment Family 1: Replication and Instrumentation

Goal: establish a reliable baseline before looking for failures.

Procedure:

1. Fit or load a J-lens for open-weight decoder models using the Anthropic
   reference implementation.
2. Reproduce basic readout examples:
   - hidden intermediate concepts in multi-hop questions
   - silent arithmetic while copying text
   - code bug detection
   - prompt-injection recognition
   - simple concept swaps such as sport or country swaps
3. Record lens logits, top-k tokens, model logits, residual activations, and
   output distributions at each tested layer and position.
4. Build a standard patching harness for activation replacement, projection,
   addition, subtraction, and constrained nullspace perturbations.

Models:

- Qwen-family model supported by the reference implementation
- Gemma or Llama-family decoder as a second architecture
- Smaller models for fast search, larger models for confirmation

Success criteria:

- Reproduce qualitative examples from the paper on at least one open model.
- Establish stable readout metrics across seeds and prompt paraphrases.
- Produce a reusable dataset of activations, J-lens readouts, and outputs.

## Experiment Family 2: J-Space Collision Search

Goal: find pairs of prompts or activations with nearly identical J-lens readouts
but different behavior.

Procedure:

1. Define a J-distance metric:
   - cosine distance between full lens-logit vectors
   - KL divergence over softmaxed lens logits
   - rank-biased overlap of top-k tokens
   - semantic distance between decoded top-k token sets
2. Define a behavior-distance metric:
   - answer mismatch
   - output distribution divergence
   - task-score difference
   - classifier difference for safety-relevant categories
3. Generate candidate prompt pairs from templates:
   - role reversals
   - negation flips
   - causal direction flips
   - ambiguous words
   - multi-token latent concepts
   - same entities, different relations
4. Search for pairs with low J-distance and high behavior-distance.
5. Confirm by patching activations across the pair and measuring whether the
   downstream behavior follows J-space or follows non-J-space context.

Example templates:

- "The person Alice trusted was Bob" vs "The person Bob trusted was Alice"
- "The country France traded with China" vs "The country China traded with
  France"
- "The program detected an exploit" vs "The program created an exploit"
- "The search result warns about prompt injection" vs "The search result is a
  prompt injection"

Primary outcome:

- Collision rate as a function of task class, layer, position, and model size.

Strong evidence against global faithfulness:

- Many low-J-distance pairs show large behavior-distance.
- Patching J-space coordinates alone fails to transfer behavior across those
  pairs.
- Auxiliary non-J-space features predict the behavioral difference.

## Experiment Family 3: Fiber Mapping

Goal: directly explore the set of activation states that share a fixed J-lens
readout.

Procedure:

1. Choose a source activation h with a clear J-space signature.
2. Compute the subspace of perturbations that minimally change L(h).
3. Sample or optimize perturbations in this approximate fiber.
4. Patch h + delta into the model.
5. Measure changes in:
   - final answer
   - next-token distribution
   - multi-step reasoning success
   - safety-relevant behavior
   - auxiliary internal features

Methods:

- Random nullspace sampling under norm constraints
- Gradient-based maximization of behavior change subject to J-distance <= epsilon
- Sparse feature perturbations constrained to preserve top-k J-lens readout
- Layer-by-layer comparison of fiber sensitivity

Primary metric:

    fiber_behavior_variance = Var(B(h + delta) | L(h + delta) ~= L(h))

Interpretation:

- Low variance supports J-space as a behaviorally sufficient coordinate system
  for that task.
- High variance shows that J-space is hiding behaviorally relevant degrees of
  freedom.

## Experiment Family 4: Same J-Space, Different Computation

Goal: test whether the same J-space content can be used differently depending
on non-J-space context.

Procedure:

1. Construct tasks where the same concept must support different computations:
   - identify language vs continue in that language
   - report country vs use country in a geographic relation
   - detect bug vs repair bug
   - notice deception vs decide whether to trust a source
2. Match or patch J-space content across tasks.
3. Measure whether behavior follows the patched J-space concept in all tasks or
   only in report-like tasks.

Connection to Anthropic claims:

Anthropic reports that swapping "Spanish" to "French" can affect naming or
reasoning about the language while leaving fluent continuation in Spanish
intact. This suggests a split between workspace-mediated and automatic
processing. This experiment generalizes that split.

Primary outcome:

- A taxonomy of operations that consult J-space versus operations that bypass it.

## Experiment Family 5: Binding and Role Structure

Goal: determine whether J-space stores merely concepts or also relations among
concepts.

Procedure:

1. Build minimal pairs with identical entity sets and different bindings.
2. Read J-space at the same token positions.
3. Compare top-k token overlap and full lens-logit similarity.
4. Test interventions:
   - swap one entity
   - swap a relation token
   - ablate a relation token
   - preserve token set while reversing roles
5. Compare against relation-sensitive probes or attribution graphs.

Task examples:

- "Alice gave Bob the key. Who has the key?"
- "Bob gave Alice the key. Who has the key?"
- "The red block is left of the blue block. What is right of the red block?"
- "The blue block is left of the red block. What is right of the blue block?"

Expected failure mode:

The readout may contain "Alice", "Bob", "key", and "gave" in both cases but not
encode who gave what to whom in a directly readable way.

Deliverable:

- A binding benchmark for J-space readouts and interventions.

## Experiment Family 6: Polysemy and Semantic Folding

Goal: test whether a single token direction conflates multiple meanings.

Procedure:

1. Select polysemous tokens:
   - bank
   - charge
   - bug
   - plant
   - draft
   - proof
2. Create prompts where the intended sense differs.
3. Compare J-lens readouts and behavior under same-token interventions.
4. Test whether adding context-specific lens vectors separates senses.

Questions:

- Does the "bug" direction mean insect, software defect, surveillance device, or
  all of them depending on context?
- Does swapping in "bank" steer toward finance in one context and riverside
  geography in another?
- Are sense distinctions stored outside the single-token J-space direction?

Success criterion:

- A sense-separation method that predicts when token-level J-space directions
  are safe to interpret literally.

## Experiment Family 7: Multi-Token and Abstract Concept Coverage

Goal: measure what the base J-lens cannot name.

Procedure:

1. Build a benchmark of latent intermediate concepts by token length:
   - single-token concepts
   - two-token names
   - technical phrases
   - abstract relations
   - idioms
2. Compare:
   - base J-lens
   - template lens
   - oracle lens
   - SAE feature labels
   - natural-language autoencoder style decoders, if available
3. Evaluate both readout and causal swap success.

Metrics:

- top-k concept recovery
- intervention success rate
- false positive rate
- semantic specificity
- robustness to paraphrase

Expected result:

Base J-lens should degrade sharply as latent concepts become multi-token or
relational. Template and oracle lenses should recover some of this lost
coverage, while introducing their own risks of premature answer readout or
confabulated phrase labels.

## Experiment Family 8: Context-Specific Jacobians

Goal: test whether the average Jacobian used by the lens hides important
contextual variation.

Procedure:

1. Fit or approximate lenses on different corpora:
   - generic web text
   - code
   - math
   - safety-evaluation transcripts
   - multilingual text
   - adversarial instructions
2. Compare readout quality across in-domain and out-of-domain prompts.
3. Compute context-local Jacobian approximations for selected prompts and
   compare them to the corpus-average lens.
4. Evaluate whether context-specific lenses reduce collision rates.

Primary question:

Is J-space one stable global coordinate frame, or a family of locally useful
frames whose average works only because many ordinary contexts are similar?

Strong finding:

- If context-specific lenses outperform the global lens on collision-heavy
  domains, then "J-space" should be treated less as one canonical object and
  more as a family of related projections.

## Experiment Family 9: Null and Shuffled Controls

Goal: ensure the method is not over-interpreting geometric artifacts.

Procedure:

1. Fit control lenses with:
   - shuffled token labels
   - shuffled target positions
   - random orthogonal transports
   - mismatched model/lens pairs
   - corrupted fitting corpora
2. Apply the same readout and intervention evaluations.
3. Measure whether controls produce plausible-looking but non-causal readouts.

Red flags:

- Shuffled lenses recover apparently meaningful concepts above chance.
- Control interventions steer outputs at nontrivial rates.
- Human raters perceive semantic coherence in random or mislabeled readouts.

Deliverable:

- A calibration suite that reports false discovery risk for any proposed lens.

## Experiment Family 10: Safety Monitor Stress Tests

Goal: test whether J-space monitoring is robust enough for alignment auditing.

Procedure:

1. Define safety-relevant latent states:
   - noticing evaluation context
   - planning deception
   - intending to fabricate evidence
   - recognizing prompt injection
   - suppressing a known policy-relevant fact
2. Build benign, controlled prompts that elicit or simulate these latent states
   without asking the model to perform harmful acts.
3. Evaluate whether J-space monitors detect the latent state before output.
4. Search for semantically equivalent states that evade token-level monitors.
5. Compare token monitors with richer monitors:
   - phrase lenses
   - SAE features
   - attribution graph features
   - supervised probes over residual activations

Primary metric:

    monitor_coverage = detected_relevant_states / known_relevant_states

Secondary metric:

    evasion_gap = behaviorally_relevant_states_missed_by_J_space_monitor

Interpretation:

- High coverage supports J-space monitoring as useful.
- A large evasion gap means J-space should be one channel in a defense-in-depth
  monitoring stack, not the whole monitor.

## Experiment Family 11: Cross-Model and Training-Stage Comparison

Goal: test whether collision patterns are model-specific or general.

Procedure:

1. Run the same battery on several open-weight models.
2. Compare base models and instruction-tuned models when possible.
3. Track whether post-training changes:
   - J-space reportability
   - self-monitoring signatures
   - collision frequency
   - safety-monitor coverage
   - context dependence

Questions:

- Does instruction tuning make J-space more report-like and less predictive of
  raw next-token continuation?
- Do larger models have lower collision rates because concepts are cleaner, or
  higher collision rates because representations are richer?
- Does model family architecture affect how globally meaningful the lens is?

Deliverable:

- A cross-model "J-space globality" leaderboard with both positive capability
  metrics and failure metrics.

## Experiment Family 12: Improved Coordinate Systems

Goal: build lenses that capture what token-level J-space misses.

Candidate extensions:

1. Phrase lens:
   - extend from token directions to phrase directions
   - benchmark on multi-token intermediates
2. Relation lens:
   - learn directions for typed relations such as subject, object, ownership,
     location, cause, and negation
   - test role-reversal tasks
3. Fiber-aware lens:
   - explicitly penalize high behavior variance within a readout fiber
   - optimize readouts for behavioral sufficiency, not just verbalizability
4. Hybrid lens:
   - combine J-lens directions with SAE features and attribution graph nodes
   - use J-lens for reportable concepts and auxiliary features for binding
5. Context-conditioned lens:
   - condition the readout map on local prompt domain or layer state
   - reduce failures caused by using a single average Jacobian

Evaluation:

Any improved lens should be judged not only on prettier readouts, but on:

- lower collision rate
- higher intervention success
- better behavior prediction
- lower false positive rate
- clearer failure boundaries

## Milestones

### Month 1: Baseline

- Install and run the reference implementation.
- Reproduce basic J-lens examples on one open model.
- Define readout, behavior, and auxiliary-state metrics.
- Produce first activation dataset.

### Month 2: Collision Search

- Build prompt-pair collision benchmark.
- Implement J-distance and behavior-distance search.
- Report first same-J, different-behavior examples.

### Month 3: Fiber Mapping

- Implement constrained nullspace perturbations.
- Estimate fiber behavior variance across layers and tasks.
- Identify layers where J-space is most and least behaviorally sufficient.

### Month 4: Binding and Polysemy

- Build relation/binding benchmark.
- Build polysemy benchmark.
- Compare token J-lens to phrase/template readouts.

### Month 5: Safety Stress Tests

- Construct controlled safety-relevant latent-state prompts.
- Measure monitor coverage and evasion gap.
- Compare J-lens monitor with richer readouts.

### Month 6: Synthesis and Improved Lenses

- Prototype one or two improved lenses.
- Re-run the hardest failure cases.
- Publish a report with benchmarks, code, and recommended interpretation
  standards.

## Key Metrics

Readout metrics:

- top-k token overlap
- rank-biased overlap
- cosine similarity of lens logits
- KL divergence of lens distributions
- semantic similarity of decoded concepts

Behavior metrics:

- exact answer match
- answer distribution divergence
- benchmark score difference
- intervention effect size
- refusal or compliance class
- safety-monitor detection class

Causal metrics:

- swap success rate
- ablation damage
- mediated-effect fraction through J-space
- patch transfer success
- fiber behavior variance

Robustness metrics:

- paraphrase stability
- seed stability
- layer stability
- model-family stability
- domain-shift degradation
- null-lens false positive rate

## Decision Criteria

Evidence supporting strong J-space faithfulness:

- Low collision rate across behaviorally relevant states.
- Low fiber behavior variance.
- J-space swaps transfer reliably across contexts.
- Non-J-space perturbations rarely change behavior when J-space is held fixed.
- Safety monitors over J-space have high coverage and low evasion gap.

Evidence against strong J-space faithfulness:

- Frequent same-J, different-behavior pairs.
- High behavior variance inside J-lens fibers.
- Binding, negation, or role structure systematically invisible to the readout.
- Context-specific lenses outperform the global average lens by large margins.
- Safety-relevant latent states evade token-level J-space monitors.

The likely outcome is intermediate:

J-space is probably a powerful interface to verbalizable cognition, but not a
complete coordinate system for model state. The research goal is to map its
domain of validity precisely.

## Practical Impact

If the program finds that J-space is globally robust, it strengthens the case
for using J-lens-derived monitors in model evaluation, safety auditing, and
debugging.

If the program finds many collisions, that is also valuable. It would clarify
that J-space readouts should be treated as lossy observables: useful, causal,
and informative, but insufficient as a complete account of hidden reasoning.

Either outcome improves the field by replacing broad claims such as "we can read
the model's thoughts" with calibrated claims:

- which thoughts
- at which layers
- for which tasks
- under which interventions
- with which blind spots

## First Concrete Study

The fastest publishable study would be:

Title:

    Hidden Folds in J-Space: Same-Readout, Different-Behavior Tests for the
    Jacobian Lens

Design:

1. Reproduce J-lens readouts on an open model.
2. Build 500 minimal prompt pairs across role reversal, negation, polysemy,
   multi-token concepts, and safety-relevant latent-state recognition.
3. Measure J-distance and behavior-distance at workspace layers.
4. Identify high-confidence collisions.
5. Patch J-space coordinates across each pair.
6. Measure whether behavior follows the J-space readout or the non-J-space
   context.
7. Release the benchmark, activation traces, and evaluation harness.

Expected contribution:

The study would not attempt to disprove J-space. It would define a sharper
standard for interpreting it: when J-lens readouts should be trusted as causal
workspace contents, and when they should be treated as local projections with
hidden structure.

