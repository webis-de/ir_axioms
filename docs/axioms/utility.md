# Utility Axioms

<p class="subtitle">Reference points, baselines, and building blocks</p>

Besides the axioms derived from the literature, `ir_axioms` includes "pseudo-axioms" that serve as reference points in experiments or as building blocks for combining axioms.

## Reference Axioms

### ORIG { #orig }

<span class="axiom">ORIG</span> `OriginalAxiom` — _"Follow the original ranking."_ [@hagen2016axiomatic]

Prefers the document with the better (lower) `rank`, or, if no ranks are given, the higher `score`.
This allows to compare (combinations of) axioms to an initial ranking, e.g., one returned by BM25, and to use the original ranking as a fallback: `axiom | ORIG()`.

### ORACLE { #oracle }

<span class="axiom">ORACLE</span> `OracleAxiom` — _"Follow the human relevance judgments."_

Prefers the document with the higher `relevance` label (e.g., from TREC qrels).
This allows to incorporate "ground-truth" relevance information into axiomatic experiments [@bondarenko2022axiomatic].
`TrecOracleAxiom` derives the same preferences directly from a `trectools` `TrecQrel` object.

For generated responses, see the [oracle axioms for RAG](generation.md#oracle-and-reference-axioms).

## Baseline Axioms

| Axiom | Class | Preference |
|:--|:--|:--|
| <span class="axiom" id="nop">NOP</span> | `NopAxiom` | Always zero (no preference). |
| <span class="axiom" id="random">RANDOM</span> | `RandomAxiom` | Random preferences (with an optional `seed`). |
| <span class="axiom" id="gt">GT</span> | `GreaterThanAxiom` | Prefer the greater output (for comparable outputs, e.g., numbers). |
| <span class="axiom" id="lt">LT</span> | `LessThanAxiom` | Prefer the smaller output. |
| <span class="axiom" id="dist">DIST</span> | `DistanceAxiom` | Signed difference of two numeric outputs. |

## Uniform and Arithmetic Axioms

The following classes implement the [operators for combining axioms](../getting-started/concepts.md#combining-axioms) and can also be instantiated directly:

| Class | Operator | Description |
|:--|:--|:--|
| `UniformAxiom(scalar=...)` | — | Constant preference (e.g., to weight other axioms). |
| `SumAxiom(axioms=[...])` | `+`, `-` | Sum of the preferences. |
| `ProductAxiom(axioms=[...])` | `*` | Product of the preferences. |
| `MultiplicativeInverseAxiom(axiom=...)` | `/` | Inverse (`1 / preference`). |
| `ConjunctionAxiom(axioms=[...])` | `&` | Preference only if all axioms agree. |
| `VoteAxiom(axioms=[...], minimum_votes=0.5)` | `%` | Vote with a minimum proportion of votes. |
| `MajorityVoteAxiom(axioms=[...], minimum_votes=0.5)` | — | Alias of `VoteAxiom` for majority votes. |
| `NegatedAxiom(axiom=...)` | unary `-` | Flipped preferences. |
| `NormalizedAxiom(axiom=...)` | unary `+` | Preferences normalized to −1, 0, 1. |

## Learned Axioms

`ScikitLearnEstimatorAxiom` combines the preferences of multiple axioms with a scikit-learn estimator (e.g., a random forest classifier) that is trained to predict the preferences of a target axiom, usually ORACLE.
In PyTerrier, use the [`EstimatorKwikSortReranker`](../walkthroughs/kwiksort-learned.md) to train such an axiom and re-rank with it.

## Wrapper Axioms

| Class | Method | Description |
|:--|:--|:--|
| `CachedAxiom` | `axiom.cached(path)` | Cache preferences on disk. |
| `ParallelAxiom` | `axiom.parallel(n_jobs)` | Compute preference matrices in parallel. |
