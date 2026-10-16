# Integrations

<p class="subtitle">PyTerrier, Pyserini, and ir_datasets</p>

Many retrieval axioms need more than the query and document texts, for example, inverse document frequencies from a retrieval index or the same tokenization as the index.
The integrations bind the [tools](concepts.md#tools-and-dependency-injection) of `ir_axioms` to a retrieval index.

## PyTerrier

Install the `pyterrier` extra:

```shell
pip install "ir_axioms[pyterrier]"
```

### Using a Terrier Index

Call `inject_pyterrier()` once before instantiating axioms, to let `ir_axioms` infer the tokenizer, index statistics, and document texts from a Terrier index:

```python
from pyterrier.datasets import get_dataset
from ir_axioms.integrations import inject_pyterrier

dataset = get_dataset("irds:msmarco-passage/trec-dl-2019")

inject_pyterrier(
    index_location="./msmarco-index/",  # Path or Terrier index (reference).
    dataset="msmarco-passage",  # Optional: load query/document texts from ir_datasets.
    text_field="text",  # Optional: document text column.
)
```

### PyTerrier Transformers

The PyTerrier integration provides the following transformers (see the [API reference](../api/pyterrier.md)):

| Transformer | Type | Description |
|:--|:--|:--|
| `KwikSortReranker` | 𝑅 → 𝑅′ | Re-rank using an axiom's preferences aggregated by KwikSort [@ailon2008aggregating]. |
| `EstimatorKwikSortReranker` | 𝑅 → 𝑅′ | Train an estimator to predict ORACLE preferences from multiple axioms, then use it to re-rank with KwikSort. |
| `AxiomaticPreferences` | 𝑅 → (𝑅×𝑅)<sub>𝑓</sub> | Compute all pairwise preferences of the given axioms. |
| `AggregatedAxiomaticPreferences` | 𝑅 → 𝑅<sub>𝑓</sub> | Aggregate the axioms' preferences for each document (e.g., as features for learning to rank). |

For example, to re-rank the top-10 BM25 results with a majority vote of retrieval axioms:

```python
from pyterrier.terrier import Retriever
from ir_axioms.axiom import MajorityVoteAxiom, ORIG, TFC1, LB1, PROX1, PROX2
from ir_axioms.integrations import KwikSortReranker

bm25 = Retriever("./msmarco-index/", wmodel="BM25")
axiom = MajorityVoteAxiom(axioms=[TFC1(), LB1(), PROX1(), PROX2()], minimum_votes=0.5) | ORIG()
pipeline = (bm25 % 10 >> KwikSortReranker(axiom=axiom)) ^ bm25
```

### Axiomatic Experiments

Similar to PyTerrier's `Experiment`, the `AxiomaticExperiment` class runs post-hoc axiomatic analyses of retrieval systems:

```python
from ir_axioms.integrations import AxiomaticExperiment

experiment = AxiomaticExperiment(
    retrieval_systems=[bm25],
    names=["BM25"],
    topics=dataset.get_topics(),
    qrels=dataset.get_qrels(),
    axioms=[TFC1(), LB1(), PROX1(), PROX2()],
    axiom_names=["TFC1", "LB1", "PROX1", "PROX2"],
    depth=10,
)
experiment.preferences  # All pairwise preferences.
experiment.preference_distribution  # How often axioms agree/disagree with ORIG.
experiment.preference_consistency  # Consistency with ORIG and ORACLE.
experiment.inconsistent_pairs  # Pairs where ORIG contradicts ORACLE.
```

The [post-hoc analysis walkthrough](../walkthroughs/post-hoc-analysis.md) explains the outputs in detail.

## Pyserini

Install the `pyserini` extra:

```shell
pip install "ir_axioms[pyserini]"
```

Then bind the tools to a Lucene index built with Pyserini/Anserini:

```python
from ir_axioms.integrations import inject_pyserini

inject_pyserini(
    index_dir="./lucene-index/",
    text_contents=True,  # Optional: read document texts from the index.
    dataset="msmarco-passage",  # Optional: load query/document texts from ir_datasets.
)
```

Afterwards, all axioms use the index's analyzer and statistics (and, optionally, its stored document contents).
Use the axioms directly via the [`Axiom` interface](concepts.md#axioms-and-preferences), e.g., `axiom.preferences(input=query, outputs=documents)`, as there are no Pyserini-specific transformers.

## ir_datasets

Queries and documents can be loaded from [`ir_datasets`](https://ir-datasets.com/), so that only IDs need to be passed to the axioms.
Pass an `ir_datasets` ID (e.g., `"msmarco-passage"`) or a loaded `ir_datasets` dataset via the `dataset` parameter of `inject_pyterrier()` or `inject_pyserini()`.
