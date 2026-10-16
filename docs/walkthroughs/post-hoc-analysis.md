# Post-Hoc Analysis

<p class="subtitle">Analyze rankings and relevance judgments with axioms</p>

This walkthrough shows how `ir_axioms` allows the post-hoc analysis of rankings and relevance judgments, using a BM25 run on the TREC 2020 Deep Learning passage ranking task.
We calculate the distribution of axiom preferences in the rankings and evaluate the consistency of the rankings and relevance judgments with retrieval axioms.

Based on the notebook [`pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb)
[![Open in Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/experiments/pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb).

## Setup

Follow the [common setup](index.md#common-setup) to load the datasets, build the index, define the `bm25` baseline, and connect `ir_axioms` to the index.

## Define the Axioms

We analyze a broad set of retrieval axioms, each cached on disk:

```python
from ir_axioms.axiom import (
    ArgUC, QTArg, QTPArg, aSL,
    LNC1, TF_LNC, LB1,
    PROX1, PROX2, PROX3, PROX4, PROX5,
    REG, ANTI_REG, ASPECT_REG,
    AND, LEN_AND, M_AND, LEN_M_AND, DIV, LEN_DIV,
    TFC1, TFC3, M_TDC, LEN_M_TDC,
    STMC1, STMC2,
)

axioms = {
    "ArgUC": ArgUC(), "QTArg": QTArg(), "QTPArg": QTPArg(), "aSL": aSL(),
    "LNC1": LNC1(), "TF_LNC": TF_LNC(), "LB1": LB1(),
    "PROX1": PROX1(), "PROX2": PROX2(), "PROX3": PROX3(), "PROX4": PROX4(), "PROX5": PROX5(),
    "REG": REG(), "ANTI_REG": ANTI_REG(), "ASPECT_REG": ASPECT_REG(),
    "AND": AND(), "LEN_AND": LEN_AND(), "M_AND": M_AND(), "LEN_M_AND": LEN_M_AND(),
    "DIV": DIV(), "LEN_DIV": LEN_DIV(),
    "TFC1": TFC1(), "TFC3": TFC3(), "M_TDC": M_TDC(), "LEN_M_TDC": LEN_M_TDC(),
    "STMC1": STMC1(), "STMC2": STMC2(),
}
axiom_names = list(axioms.keys())
axioms = [axiom.cached(cache_dir / "axioms" / name) for name, axiom in axioms.items()]
```

## Axiomatic Experiments

The [`AxiomaticExperiment`][ir_axioms.integrations.pyterrier.experiment.AxiomaticExperiment] class is the entry point for post-hoc analyses of rankings and relevance judgments.
We pass it the retrieval pipelines to evaluate, the topics and qrels, and the axioms to compute preferences for.
The `depth` parameter controls up to which rank the preferences are computed.

Computing time roughly scales with \(|\text{systems}| \times |\text{topics}| \times \text{depth}^2 \times |\text{axioms}|\).
Additional parameters help to reduce computational costs by filtering topics without qrels (`filter_by_topics`) and documents without qrels (`filter_by_qrels`).
In our experience, caching helps most when analyzing multiple runs on the same benchmark (like TREC runs), because the result sets of different runs often overlap.

```python
from ir_axioms.integrations import AxiomaticExperiment

experiment = AxiomaticExperiment(
    retrieval_systems=[bm25],
    names=["BM25"],
    topics=dataset_test.get_topics(),
    qrels=dataset_test.get_qrels(),
    axioms=axioms,
    axiom_names=axiom_names,
    depth=10,
    filter_by_qrels=False,
    filter_by_topics=False,
    verbose=True,
)
```

The experiment automatically adds the [ORIG](../axioms/utility.md#orig) and [ORACLE](../axioms/utility.md#oracle) pseudo-axioms, whose preferences follow the original ranking and the relevance judgments, respectively.

### All Preferences

The `preferences` property returns a data frame with all pairwise preferences up to the specified depth.
It has one row per retrieval system, query, and document pair (`docno_a`, `docno_b`), and one column `<axiom>_preference` per axiom:

```python
experiment.preferences
```

### Inconsistent Document Pairs

Investigate document pairs that the retrieval system ranked in the wrong order according to the relevance judgments (i.e., ORIG contradicts ORACLE).
The axioms' preferences for these pairs can provide insights to improve the evaluated ranking function:

```python
experiment.inconsistent_pairs
```

### Consistency of Runs with Axioms

This evaluation shows how consistent the axioms are with the run's original order (ORIG) and with the relevance judgments (ORACLE):

- High consistency with ORACLE indicates that the axiom captures the relevance judgments well.
- High consistency with ORIG indicates that the run adheres to the axiom's constraint well.

```python
experiment.preference_consistency
```

### Preference Distribution

The distribution of each axiom's preferences compared to the original ranking (no preference, same as ORIG, opposite of ORIG) helps to identify decisive axioms for your dataset:

```python
experiment.preference_distribution
```

## Parallelization

For larger experiments, compute the preferences in parallel, using either [Joblib](https://joblib.readthedocs.io/) (default) or [Ray](https://www.ray.io/) as backend:

```python
experiment = AxiomaticExperiment(
    ...,
    parallel_jobs=4,
    parallel_backend="joblib",
)
```
