# Hands-on Tutorial

<p class="subtitle">Analyze retrieval systems, spot errors, and fix them by axiomatic re-ranking</p>

In this tutorial, we demonstrate how to:

- run axiomatic analyses on existing retrieval models,
- axiomatically re-rank retrieval results, and
- combine axioms into a new, custom axiom.

Based on the notebook [`grenoble2025_showcase.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/grenoble2025_showcase.ipynb).

## Data and Index

We use the TREC 2019 Deep Learning passage ranking data from `ir_datasets`:

```python
from ir_datasets import load
from pyterrier.datasets import get_dataset

dataset = load("msmarco-passage/trec-dl-2019")
pt_dataset = get_dataset("irds:msmarco-passage/trec-dl-2019")
```

Then, we index all MS MARCO passages (skipping indexing if the index already exists):

```python
from pathlib import Path
from pyterrier import IndexRef
from pyterrier.terrier import IterDictIndexer

index_dir = Path("msmarco-index/")
if not index_dir.exists():
    # Takes about 10–15 minutes from scratch.
    index = IterDictIndexer(str(index_dir.absolute())).index(pt_dataset.get_corpus_iter())
else:
    index = IndexRef.of(str(index_dir.absolute()))
```

## Retrieval Models

Next, we define two retrieval models with [PyTerrier](https://pyterrier.readthedocs.io/), BM25 and a language model (Hiemstra), and retrieve the top-10 results each:

```python
from pyterrier.terrier import Retriever

bm25 = Retriever(index, wmodel="BM25", num_results=10)
lm = Retriever(index, wmodel="Hiemstra_LM", num_results=10)

bm25.search("test").head()
```

## Connect `ir_axioms` to the Index

Let `ir_axioms` infer text and index statistics from the Terrier index:

```python
from ir_axioms.integrations import inject_pyterrier

inject_pyterrier(
    index_location=index,
    dataset=dataset,
)
```

## Post-hoc Axiomatic Analyses

Similar to PyTerrier's `Experiment`, `ir_axioms` runs axiomatic analyses with the `AxiomaticExperiment` class.
In the constructor, specify:

- the retrieval systems to test (and their names),
- the axioms to consider (and their names),
- the maximum ranking depth (top-\(k\) for building preference matrices), and
- the topics and qrels.

```python
from ir_axioms.axiom import TFC1, LB1, PROX1, PROX2
from ir_axioms.integrations import AxiomaticExperiment

experiment = AxiomaticExperiment(
    retrieval_systems=[bm25, lm],
    names=["BM25", "LM"],
    axioms=[TFC1(), LB1(), PROX1(), PROX2()],
    axiom_names=["TFC1", "LB1", "PROX1", "PROX2"],
    depth=5,
    topics=pt_dataset.get_topics()[:50],
    qrels=pt_dataset.get_qrels(),
    filter_by_qrels=True,
)
```

This does not execute any experiment just yet, but prepares the full configuration.
The results are computed lazily when accessing the experiment's properties.

### Pairwise Preferences

`experiment.preferences` returns a data frame with one row per pair of documents retrieved in the top-\(k\), and one column per axiom.
Besides the given axioms, the [ORIG](../axioms/utility.md#orig) and [ORACLE](../axioms/utility.md#oracle) pseudo-axioms are added automatically:

```python
df = experiment.preferences
df[df["LB1_preference"] != 0]  # Only pairs where LB1 has a preference.

experiment.preferences[
    ["name", "query", "docno_a", "docno_b", "ORIG_preference", "ORACLE_preference", "TFC1_preference"]
].head()
```

### Preference Distribution

Next, we aggregate the axiom preferences and count for each axiom how often:

- a "zero" preference is returned (i.e., no preference),
- the axiom agrees with ORIG, and
- the axiom disagrees with ORIG.

This helps to quickly identify "useful" or "interesting" axioms, e.g., axioms that often disagree with ORIG:

```python
experiment.preference_distribution
experiment.preference_consistency.round(2)
```

### Inconsistent Pairs

Looking at inconsistent pairs of documents (i.e., pairs that the retrieval system ranked in the wrong order, where ORIG contradicts ORACLE) helps to understand the problem better.
The `<axiom>_preference` columns then show which axioms would have corrected the mistake:

```python
inconsistent_pairs = experiment.inconsistent_pairs
inconsistent_pairs = inconsistent_pairs[inconsistent_pairs["label_b"] == 0]
inconsistent_pair = inconsistent_pairs.head(10)
```

Using `ir_datasets`, we can easily inspect the document texts and their relevance labels:

```python
docs_store = dataset.docs_store()
docs_store.get(inconsistent_pair["docno_a"].iloc[0]).text, inconsistent_pair["label_a"].iloc[0]
docs_store.get(inconsistent_pair["docno_b"].iloc[0]).text, inconsistent_pair["label_b"].iloc[0]
```

## Axiomatic Re-Ranking

Now that we have spotted an error, let's fix it by axiomatic re-ranking: we first combine multiple axioms in a majority vote, and then re-rank based on the majority's preferences.

### Majority Vote

The idea is to change the ranking order only where multiple axioms agree, and to fall back to the original ranking order otherwise.
The combined axiom is itself an `Axiom`:

```python
from ir_axioms.axiom import ORIG, MajorityVoteAxiom

majority_vote = MajorityVoteAxiom(
    axioms=[TFC1(), LB1(), PROX1(), PROX2()],
    minimum_votes=0.5,
) | ORIG()
```

### KwikSort

Based on the majority vote, we can apply simple algorithms like KwikSort [@ailon2008aggregating].
KwikSort is similar to Quicksort:

1. Select a pivot.
2. Compare all other documents to the pivot and split them into documents ranked before and after the pivot.
3. Repeat for both sub-rankings.

Note that the results depend on the pivot selection.
KwikSort is implemented as a PyTerrier `Transformer`:

```python
from ir_axioms.integrations import KwikSortReranker

kwiksort = KwikSortReranker(axiom=majority_vote, verbose=True)
bm25_kwiksort = (bm25 % 10 >> kwiksort) ^ bm25
lm_kwiksort = (lm % 10 >> kwiksort) ^ lm
```

### Evaluate the Re-Rankers

To verify the effectiveness of the axiomatic re-ranking, we run a PyTerrier `Experiment` comparing the two retrieval systems and their re-ranked variants:

```python
from pyterrier import Experiment

Experiment(
    [bm25, bm25_kwiksort, lm, lm_kwiksort],
    pt_dataset.get_topics(),
    pt_dataset.get_qrels(),
    eval_metrics=["ndcg_cut_10"],
    names=["BM25", "BM25 + KwikSort", "LM", "LM + KwikSort"],
)
```

!!! success "Results"

    In the original showcase, axiomatic re-ranking only slightly changes the effectiveness, but the rankings are now more consistent with the axioms.

    Potential improvements:

    - Instead of a majority vote, train a classifier to weight the axioms (see [learning the ORACLE](kwiksort-learned.md)).
    - Choose axioms tailored to the use case, instead of a few exemplary axioms.
