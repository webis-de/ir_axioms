# Core Concepts

<p class="subtitle">Axioms, preferences, combinations, and tools</p>

## Axioms and Preferences

An **axiom** describes a pairwise constraint between two outputs given the same input, expressed as a **preference**.
In `ir_axioms`, every axiom extends the generic abstract base class [`Axiom[Input, Output]`](../api/axiom.md):

- Retrieval axioms compare two documents for a query: `Axiom[Query, Document]`.
- RAG axioms compare two generated responses for a request: `Axiom[GenerationRequest, GenerationResponse]`.
- Utility axioms (e.g., [`ORIG`](../axioms/utility.md#orig) or [`RANDOM`](../axioms/utility.md#random)) work with any input/output types.

The preference \(\mathrm{pref}_A(q, d_i, d_j)\) of an axiom \(A\) for an input \(q\) and outputs \(d_i\), \(d_j\) is a float with the following semantics [@bondarenko2022axiomatic]:

\[
\begin{aligned}
\mathrm{pref}_A(q, d_i, d_j) > 0 &\iff d_i >_A d_j, \\
\mathrm{pref}_A(q, d_i, d_j) < 0 &\iff d_i <_A d_j, \\
\mathrm{pref}_A(q, d_i, d_j) = 0 &\iff d_i \not>_A d_j \land d_i \not<_A d_j.
\end{aligned}
\]

A zero preference is returned when an axiom does not express a preference, for example, if the axiom's preconditions are not met.
Preferences should be antisymmetric: flipping the outputs flips the sign of the preference.

Each axiom implements up to three methods:

| Method | Description |
|:--|:--|
| `preference(input, output1, output2)` | Pairwise preference between two outputs (must be implemented). |
| `preferences(input, outputs)` | Preference matrix for a sequence of outputs. The default implementation calls `preference()` for all pairs, but axioms can override it to compute costly features only once per output. |
| `batch_preferences(batch)` | Stacked preference matrices for multiple inputs (e.g., to batch model inference across queries). |

## Instantiating Axioms

Each axiom class is also available under a short alias, e.g., `TFC1` for `Tfc1Axiom`.
The aliases are created with `lazy_inject()` and resolve all dependencies of the axiom (e.g., tokenizers or index statistics) via [dependency injection](#tools-and-dependency-injection) when called:

```python
from ir_axioms.axiom import TFC1, Tfc1Axiom

axiom = TFC1()  # Dependencies are injected automatically.
axiom = TFC1(margin_fraction=0.2)  # Override parameters as keyword arguments.
```

Most axioms expose parameters to relax or strengthen their conditions, most notably `margin_fraction`, the relative tolerance below which two values (e.g., term frequencies) are considered equal.
The [axiom catalog](../axioms/index.md) lists the parameters of each axiom.

## Combining Axioms

Axioms can be combined with Python operators.
The result of a combination is again an `Axiom`, so combinations can be nested arbitrarily.

| Operator | Result | Description |
|:--|:--|:--|
| `a + b` | `SumAxiom` | Sum of the preferences. |
| `a - b` | `SumAxiom` | Difference of the preferences (`a + -b`). |
| `a * b` | `ProductAxiom` | Product of the preferences. |
| `a / b` | `ProductAxiom` | Product with the multiplicative inverse of `b`'s preferences. |
| `a % b` | `VoteAxiom` | Majority vote (absolute majority, i.e., at least 50 % of the votes). |
| `a & b` | `ConjunctionAxiom` | Preference only if all axioms agree, otherwise zero. |
| `a \| b` | `CascadeAxiom` | First non-zero preference (i.e., `b` is the fallback when `a` has no preference). |
| `-a` | `NegatedAxiom` | Flip the sign of the preferences. |
| `+a` | `NormalizedAxiom` | Normalize the preferences to −1, 0, or 1 (same as `a.normalized()`). |

To weight an axiom by a constant, multiply with a [`UniformAxiom`](../axioms/utility.md#uniform-and-arithmetic-axioms):

```python
from ir_axioms.axiom import TFC1, TFC3, UniformAxiom

# Linear combination of TFC1 and TFC3.
axiom = TFC1() + UniformAxiom(scalar=2.0) * TFC3()
```

To control the required majority of a vote, instantiate the `VoteAxiom` (or `MajorityVoteAxiom`) directly:

```python
from ir_axioms.axiom import MajorityVoteAxiom, ORIG, TFC1, LB1, PROX1, PROX2

axiom = MajorityVoteAxiom(
    axioms=[TFC1(), LB1(), PROX1(), PROX2()],
    minimum_votes=0.5,  # 0.5 for absolute majority, 0 for relative majority, 1 for consensus.
) | ORIG()
```

!!! warning "Mathematical properties of combined axioms"

    Products and multiplicative inverses do not necessarily preserve asymmetry, irreflexivity, or transitivity of the preferences.
    For example, for `axiom * axiom`, any negative preference's sign is flipped.

## Caching and Parallelization

Some axioms are expensive to compute (e.g., axioms calling external APIs or neural models).
Cache an axiom's preferences on disk with `cached()`, so that each input–outputs tuple is computed only once:

```python
from pathlib import Path
from ir_axioms.axiom import ArgUC

axiom = ArgUC().cached(Path("cache/ArgUC"))
```

!!! note

    There is no implicit caching in `ir_axioms` ≥ 1.0. You have to choose the cache location explicitly.

To compute preference matrices in parallel, use `parallel()`:

```python
axiom = ArgUC().parallel(n_jobs=4)
```

## Preconditions

Several retrieval axioms only express a preference if some precondition holds, for example, if both documents have (approximately) the same length.
These axioms extend the [`PreconditionMixin`](../api/axiom.md) and accept a `precondition` parameter:

```python
from ir_axioms.axiom import LEN_AND
from ir_axioms.precondition import LEN, NOP

LEN_AND()  # Default: length precondition with 10 % margin (LEN).
LEN_AND(precondition=LEN(margin_fraction=0.2))  # Relax the length precondition.
LEN_AND(precondition=NOP())  # Disable the precondition.
```

Any axiom can also serve as a precondition via `axiom.as_precondition()`.

## Tools and Dependency Injection

Axioms rely on _tools_ like tokenizers, text statistics, or similarity measures.
These tools are declared as dataclass fields with interface types and are injected automatically using the [`injector`](https://injector.readthedocs.io/) library.
This way, the same axiom can, for example, use term statistics from a Terrier index or from a Pyserini index, depending on the [integration](integrations.md) you use.

The default `ToolsModule` binds the following tool interfaces.
All bindings use the `singleton` scope, i.e., each interface resolves to the same instance within one `Injector`.

| Interface | Default implementation |
|:--|:--|
| `AspectExtraction` | `SpacyNounChunksAspectExtraction` |
| `PivotSelection` | `RandomPivotSelection` |
| `TermTokenizer` | `BlingfireTermTokenizer` (if Blingfire installed) or `SpacyTermTokenizer` |
| `SentenceTokenizer` | `BlingfireSentenceTokenizer` (if Blingfire installed) or `SpacySentenceTokenizer` |
| `TermSimilarity` | `FastTextTermSimilarity` if fastText is installed, else `WordNetSynonymSetTermSimilarity` |
| `SentenceSimilarity` | `SentenceTransformersSentenceSimilarity` (if SBERT installed) or `Model2VecSentenceSimilarity` (if model2vec installed) or `AverageTermSimilaritySentenceSimilarity` |
| `TextContents[T]` for `HasText`, `Query`, `Document`, `GenerationRequest`, `GenerationResponse` | `SimpleTextContents` |
| `TextContents[Union[Document, Query]]` | `DocumentQueryTextContents` |
| `TextContents[Union[GenerationRequest, GenerationResponse]]` | `GenerationInputOutputTextContents` |
| `TextStatistics[T]` for `Query`, `Document`, `GenerationRequest`, `GenerationResponse` | `SimpleTextStatistics` |
| `TextStatistics[Union[Document, Query]]` | `DocumentQueryTextStatistics` |
| `TextStatistics[Union[GenerationRequest, GenerationResponse]]` | `GenerationInputOutputTextStatistics` |

!!! info "Index statistics"

    Axioms that depend on collection statistics (e.g., inverse document frequencies for [TFC3](../axioms/retrieval.md#tfc3) or [M-TDC](../axioms/retrieval.md#m-tdc)) require an `IndexStatistics` binding, which is provided by the [PyTerrier or Pyserini integrations](integrations.md).

To replace a tool for a single axiom, pass it as a keyword argument:

```python
from ir_axioms.axiom import COV1
from ir_axioms.tools import SpacyEntitiesAspectExtraction

axiom = COV1(aspect_extraction=SpacyEntitiesAspectExtraction())
```
