# Migration to `ir_axioms` 1.0

If you have used `ir_axioms` <1.0 and now want to migrate your code to the new implementation, please follow this guide.

## The `Axiom` Interface

See this example on how to define an axiom in the old (<1.0) version and the recent version of `ir_axioms`.

### Old Axiom Definition

```python
class TraditionalAxiom(Axiom):
    def preference(self, index, q, d_1, d_2) -> float:
        # Preference between documents d_1 and d_2 for the query q given a retrieval index.
```

### New Axiom Definition

```python
class NewAxiom(Axiom[QueryType, ResponseType]):
    def preference(self, q, d_1, d_2) -> float:
        # Preference between responses d_1, d_2 for query q.
    def preferences(self, q, D) -> PreferenceMatrix:  # a square `ndarray`
        # (Optional) Pairwise preference matrix for a sequence D of responses for query q.
```

Key changes are:

- There is no hard-coded dependency on the retrieval index.
- You can override the batch-computation of axiomatic preferences to speed up the preference computation.

For further steps on developing new axioms, refer to our [dedicated guidelines](./axiom-development.md).

## Usage with PyTerrier

To use the axioms with [PyTerrier](https://pyterrier.readthedocs.io/en/latest/), continue to use the PyTerrier `Transformer` subclasses and the `AxiomaticExperiment` as before:

- `KwikSortReranker` to re-rank results based on a given axiom.
- `AxiomaticPreferences` to return a data frame with all pairwise preferences of the given axiom(s).
- `AggregatedAxiomaticPreferences` to return the same preferences aggregated by document (e.g., as features for LTR).
- `AxiomaticExperiment` to run an axiomatic experiment (traditional IR).

The new RAG-specific axioms can be used as shown in the [ICTIR 2025 notebook](../experiments/ictir2025.ipynb).

## Other Breaking Changes

- There is no longer implicit disk-caching of axioms. Instead, you now have to define the cache location explicitly, using the `CachedAxiom` or the `Axiom.cached()` instance method.
- Axioms can no longer be loaded by name. The class registry was removed for simplicity.
- Some version constraints on sub-dependencies (e.g., PyTerrier) have been strictened.
- Magnitude versions of word-embeddings have been replaced by [TODO][TODO]
