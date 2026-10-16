# Migrating to `ir_axioms` 1.0

<p class="subtitle">Upgrade code written for <code>ir_axioms</code> &lt; 1.0</p>

If you have used `ir_axioms` < 1.0 and now want to migrate your code, please follow this guide.

!!! note "Still need to support `ir_axioms` < 1.0?"

    The legacy version is still available on the [`legacy` branch](https://github.com/webis-de/ir_axioms/tree/legacy) and you can still install old versions:

    ```shell
    pip install ir_axioms<1.0
    ```

## The `Axiom` Interface

See this example on how to define an axiom in the old (< 1.0) version and the recent version of `ir_axioms`.

=== "Old axiom definition (< 1.0)"

    ```python
    class TraditionalAxiom(Axiom):
        def preference(self, index, q, d_1, d_2) -> float:
            # Preference between documents d_1 and d_2 for the query q given a retrieval index.
            ...
    ```

=== "New axiom definition (≥ 1.0)"

    ```python
    class NewAxiom(Axiom[QueryType, ResponseType]):
        def preference(self, q, d_1, d_2) -> float:
            # Preference between responses d_1, d_2 for query q.
            ...

        def preferences(self, q, D) -> PreferenceMatrix:  # a square `ndarray`
            # (Optional) Pairwise preference matrix for a sequence D of responses for query q.
            ...
    ```

Key changes are:

- There is no hard-coded dependency on the retrieval index.
  Instead, axioms declare the [tools](concepts.md#tools-and-dependency-injection) they need, which are injected automatically.
- You can override the batch-computation of axiomatic preferences to speed up the preference computation.

For further steps on developing new axioms, refer to our [dedicated guidelines](../development/axiom-development.md).

## Usage with PyTerrier

To use the axioms with [PyTerrier](https://pyterrier.readthedocs.io/en/latest/), continue to use the PyTerrier `Transformer` subclasses and the [`AxiomaticExperiment`][ir_axioms.integrations.pyterrier.experiment.AxiomaticExperiment] as before:

- [`KwikSortReranker`][ir_axioms.integrations.pyterrier.transformers.KwikSortReranker] to re-rank results based on a given axiom.
- [`AxiomaticPreferences`][ir_axioms.integrations.pyterrier.transformers.AxiomaticPreferences] to return a data frame with all pairwise preferences of the given axiom(s).
- [`AggregatedAxiomaticPreferences`][ir_axioms.integrations.pyterrier.transformers.AggregatedAxiomaticPreferences] to return the same preferences aggregated by document (e.g., as features for LTR).
- `AxiomaticExperiment` to run an axiomatic experiment (traditional IR).

Instead of passing the index and dataset to each transformer, call [`inject_pyterrier()`](integrations.md#using-a-terrier-index) once.

The new RAG-specific axioms can be used as shown in the [RAG evaluation walkthrough](../walkthroughs/rag-evaluation.md), which is based on the [ICTIR 2025 notebook](https://github.com/webis-de/ir_axioms/blob/main/experiments/ictir2025.ipynb).

## Other Breaking Changes

- There is no longer implicit disk-caching of axioms (formerly the `~` operator). Instead, you now have to define the cache location explicitly, using the [`CachedAxiom`][ir_axioms.axiom.cache.CachedAxiom] or the `Axiom.cached()` instance method.
- Axioms can no longer be loaded by name. The class registry was removed for simplicity.
- Some version constraints on sub-dependencies (e.g., PyTerrier) have been strictened.
- Magnitude versions of word-embeddings have been replaced by Hugging Face download and manual fastText model loading.
  The separate fastText-based axiom variants (e.g., `STMC1_f`, `REG_f`) were removed; the term similarity is now a [tool](concepts.md#tools-and-dependency-injection) that uses fastText automatically if installed.
- The Terrier- and Anserini-integration-related imports moved from `ir_axioms.backend.pyterrier` etc. to `ir_axioms.integrations.pyterrier` etc.

!!! tip "Quick reference"

    | Legacy (< 1.0) | Current (≥ 1.0) |
    |:--|:--|
    | `~axiom` | `axiom.cached(Path("cache/axiom"))` |
    | `ir_axioms.backend.pyterrier.transformers` | `ir_axioms.integrations` |
    | `ir_axioms.modules.pivot` | `ir_axioms.tools` |
    | `KwikSortReranker(axiom=..., index=..., dataset=...)` | `inject_pyterrier(index_location=..., dataset=...)`, then `KwikSortReranker(axiom=...)` |
    | `VoteAxiom([...], minimum_votes=0.5)` | `MajorityVoteAxiom(axioms=[...], minimum_votes=0.5)` |
    | `to_axiom("TFC1")`, `AutoAxiom` | `TFC1()` |
