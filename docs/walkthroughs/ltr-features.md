# Axiomatic LTR Features

<p class="subtitle">Aggregate axiom preferences as learning-to-rank features</p>

This walkthrough shows how `ir_axioms` generates features for learning-to-rank approaches like LambdaMART in PyTerrier, by aggregating the pairwise preferences of different axioms for each document.
We train a LambdaMART re-ranker with these axiomatic features on the TREC 2019 Deep Learning topics and qrels, re-rank the TREC 2020 topics, and evaluate nDCG@10, reciprocal rank, and average precision.

Based on the notebook [`pyterrier_ltr_features.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_ltr_features.ipynb)
[![Open in Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/experiments/pyterrier_ltr_features.ipynb).

## Setup

Follow the [common setup](index.md#common-setup) to load the datasets, build the index, define the `bm25` baseline, and connect `ir_axioms` to the index.

## Define the Axioms

We use the same axioms as in the [previous walkthrough](kwiksort-learned.md#define-the-axioms), cached on disk:

```python
from ir_axioms.axiom import (
    ArgUC, QTArg, QTPArg, aSL,
    LNC1, TF_LNC, LB1,
    PROX1, PROX2, PROX3, PROX4, PROX5,
    REG, ANTI_REG, ASPECT_REG,
    AND, LEN_AND, M_AND, LEN_M_AND, DIV, LEN_DIV,
    TFC1, TFC3, M_TDC, LEN_M_TDC,
    STMC1, STMC2,
    ORIG,
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
axioms = [axiom.cached(cache_dir / "axioms" / name) for name, axiom in axioms.items()]
axioms.append(ORIG())
```

## Aggregate Preferences

To reduce each document's row of the preference matrix (its preferences compared to all other documents) to a fixed number of features, we define aggregation functions.
In this example, we count how often an axiom prefers the document, has no preference, or prefers the other document:

```python
aggregations = [
    lambda prefs: sum(p > 0 for p in prefs) / len(prefs),
    lambda prefs: sum(p == 0 for p in prefs) / len(prefs),
    lambda prefs: sum(p < 0 for p in prefs) / len(prefs),
]
```

(By default, [`AggregatedAxiomaticPreferences`][ir_axioms.integrations.pyterrier.transformers.AggregatedAxiomaticPreferences] uses `max`, `min`, and `mean`.)
With the axioms and aggregation functions, we create axiomatic features for the top-20 results of the baseline ranking:

```python
from ir_axioms.integrations import AggregatedAxiomaticPreferences

features = bm25 % 20 >> AggregatedAxiomaticPreferences(
    axioms=axioms,
    aggregations=aggregations,
    verbose=True,
)
```

Each document then has \(|\text{axioms}| \times |\text{aggregations}|\) features.
For example, for the first training topic:

```python
features.transform(dataset_train.get_topics()[:1])["features"]
```

## Learning to Rank

The aggregated preference features can be used with any learning-to-rank approach.
Here, we train a LambdaMART ranker (using LightGBM) that optimizes nDCG@10:

```python
from lightgbm import LGBMRanker
from pyterrier.ltr import apply_learned_model

lambda_mart = LGBMRanker(
    num_iterations=1000,
    metric="ndcg",
    eval_at=[10],
    importance_type="gain",
)
ltr = features >> apply_learned_model(lambda_mart, form="ltr")
```

We train the ranker on the TREC 2019 topics, using the last 5 topics for validation:

```python
ltr.fit(
    dataset_train.get_topics()[:-5],
    dataset_train.get_qrels(),
    dataset_train.get_topics()[-5:],
    dataset_train.get_qrels(),
)
```

## Experimental Evaluation

```python
from ir_measures import nDCG, MAP, RR
from pyterrier import Experiment

experiment = Experiment(
    [bm25, ltr ^ bm25],
    dataset_test.get_topics(),
    dataset_test.get_qrels(),
    [nDCG @ 10, RR, MAP],
    names=["BM25", "Axiomatic LTR"],
    verbose=True,
)
experiment.sort_values(by="nDCG@10", ascending=False)
```

## Feature Importances

LambdaMART's feature importances help to identify axioms or aggregations that are not used for re-ranking:

```python
feature_importance = lambda_mart.feature_importances_.reshape(-1, len(aggregations))
feature_importance.sum(axis=0)  # Importance per aggregation.
feature_importance.sum(axis=1)  # Importance per axiom.
```
