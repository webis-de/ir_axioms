# Utility Axioms

<p class="subtitle">Reference points, baselines, and building blocks</p>

Besides the axioms derived from the literature, `ir_axioms` includes "pseudo-axioms" that serve as reference points in experiments, as baselines, or as building blocks for [combining axioms](../getting-started/concepts.md#combining-axioms).
Most of them work with any input and output types.
The [API reference](../api/axiom.md#combinators) documents all classes and parameters.

## Reference Axioms

### ORIG { #orig }

<span class="badge">ORIG</span> [`OriginalAxiom`][ir_axioms.axiom.retrieval.simple.OriginalAxiom] — _"Follow the original ranking."_ [@hagen2016axiomatic]

Implementation
:   Prefers the document with the better (lower) `rank`, or, if no ranks are given, the higher `score`.

Usage
:   Compare (combinations of) axioms to an initial ranking, e.g., one returned by BM25, or use the original ranking as a fallback: `axiom | ORIG()`.

### ORACLE { #oracle }

<span class="badge">ORACLE</span> [`OracleAxiom`][ir_axioms.axiom.retrieval.simple.OracleAxiom] — _"Follow the human relevance judgments."_ [@bondarenko2022axiomatic]

Implementation
:   Prefers the document with the higher `relevance` label (e.g., from TREC qrels).

Usage
:   Incorporate "ground-truth" relevance information into axiomatic experiments, e.g., as the target for [learning to combine axioms](../walkthroughs/kwiksort-learned.md).

Variants
:   [`TrecOracleAxiom`][ir_axioms.axiom.retrieval.trec.TrecOracleAxiom]: derives the same preferences directly from a `trectools` `TrecQrel` object.
:   For generated responses, see the [oracle axioms for RAG](generation.md#oracle-axioms).

## Baseline Axioms

### NOP { #nop }

<span class="badge">NOP</span> [`NopAxiom`][ir_axioms.axiom.simple.NopAxiom] — _"Never prefer any output."_

Implementation
:   Always returns a zero preference.

### RANDOM { #random }

<span class="badge">RANDOM</span> [`RandomAxiom`][ir_axioms.axiom.simple.RandomAxiom] — _"Prefer outputs at random."_

Implementation
:   Returns random preferences (see the [known issue](../development/roadmap.md#known-bugs)).

Parameters
:   `seed=None`.

### GT and LT { #gt }

<span class="badge">GT</span> [`GreaterThanAxiom`][ir_axioms.axiom.simple.GreaterThanAxiom] — _"Prefer the greater output."_

<span class="badge" id="lt">LT</span> [`LessThanAxiom`][ir_axioms.axiom.simple.LessThanAxiom] — _"Prefer the smaller output."_

Implementation
:   Compares the outputs directly, for any comparable outputs (e.g., numbers).

### DIST { #dist }

<span class="badge">DIST</span> [`DistanceAxiom`][ir_axioms.axiom.simple.DistanceAxiom] — _"Prefer the greater output, proportionally to the difference."_

Implementation
:   Returns the signed difference of two numeric outputs.

## Arithmetic Axioms

The following axioms implement the [operators for combining axioms](../getting-started/concepts.md#combining-axioms) and can also be instantiated directly.

### Uniform { #uniform }

[`UniformAxiom`][ir_axioms.axiom.arithmetic.UniformAxiom] — _"Return a constant preference."_

Usage
:   Weight other axioms, e.g., `TFC1() + UniformAxiom(scalar=2.0) * TFC3()`.

Parameters
:   `scalar`.

### Sum and Product { #sum-and-product }

[`SumAxiom`][ir_axioms.axiom.arithmetic.SumAxiom] (operators `+`, `-`) — _"Add the preferences."_

[`ProductAxiom`][ir_axioms.axiom.arithmetic.ProductAxiom] (operator `*`) — _"Multiply the preferences."_

[`MultiplicativeInverseAxiom`][ir_axioms.axiom.arithmetic.MultiplicativeInverseAxiom] (operator `/`) — _"Invert the preferences (`1 / preference`)."_

Parameters
:   `axioms` (or `axiom` for the inverse).

### Vote { #vote }

[`VoteAxiom`][ir_axioms.axiom.arithmetic.VoteAxiom] (operator `%`) and [`MajorityVoteAxiom`][ir_axioms.axiom.arithmetic.MajorityVoteAxiom] — _"Prefer what the majority of axioms prefers."_

Implementation
:   Counts the positive and negative preferences of the axioms and prefers the output with more votes, if these reach the minimum proportion of votes.

Parameters
:   `axioms`, `minimum_votes=0.5` (0.5 for absolute majority, 0 for relative majority, 1 for consensus).

### Conjunction and Cascade { #conjunction-and-cascade }

[`ConjunctionAxiom`][ir_axioms.axiom.arithmetic.ConjunctionAxiom] (operator `&`) — _"Prefer only if all axioms agree."_

[`CascadeAxiom`][ir_axioms.axiom.arithmetic.CascadeAxiom] (operator `|`) — _"Use the first non-zero preference."_

Parameters
:   `axioms`.

### Negation and Normalization { #negation-and-normalization }

[`NegatedAxiom`][ir_axioms.axiom.arithmetic.NegatedAxiom] (unary `-`) — _"Flip the preferences."_

[`NormalizedAxiom`][ir_axioms.axiom.arithmetic.NormalizedAxiom] (unary `+`, `axiom.normalized()`) — _"Normalize the preferences to −1, 0, or 1."_

Parameters
:   `axiom`.

## Learned Axioms

### Estimator { #estimator }

[`ScikitLearnEstimatorAxiom`][ir_axioms.axiom.estimator.ScikitLearnEstimatorAxiom] — _"Prefer what a trained classifier predicts."_

Implementation
:   Combines the preferences of multiple axioms with a scikit-learn estimator (e.g., a random forest classifier) that is trained to predict the preferences of a target axiom, usually [ORACLE](#oracle).

Usage
:   In PyTerrier, use the [`EstimatorKwikSortReranker`][ir_axioms.integrations.pyterrier.estimator.EstimatorKwikSortReranker] to train such an axiom and re-rank with it (see the [walkthrough](../walkthroughs/kwiksort-learned.md)).

Parameters
:   `axioms`, `estimator`.

## Wrapper Axioms

### Cached { #cached }

[`CachedAxiom`][ir_axioms.axiom.cache.CachedAxiom] (`axiom.cached(path)`) — _"Compute each preference only once."_

Implementation
:   Stores the preferences of the wrapped axiom in an on-disk database at the given path.

### Parallel { #parallel }

[`ParallelAxiom`][ir_axioms.axiom.parallel.ParallelAxiom] (`axiom.parallel(n_jobs)`) — _"Compute preference matrices in parallel."_

Parameters
:   `n_jobs=None` (number of parallel jobs).
