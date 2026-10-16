# Learning the ORACLE

<p class="subtitle">KwikSort re-ranking with an estimated ORACLE axiom</p>

This walkthrough shows how to re-rank results in PyTerrier with KwikSort and an _estimation_ of the [ORACLE](../axioms/utility.md#oracle) axiom.
The ORACLE axiom replicates the perfect ordering induced by human relevance judgments (i.e., qrels).
We combine the preferences of many axioms in a random forest classifier that is trained to predict ORACLE preferences on the TREC 2019 Deep Learning topics and qrels, and then re-rank the TREC 2020 topics with the trained axiom.

Based on the notebook [`pyterrier_kwiksort_learned.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_kwiksort_learned.ipynb)
[![Open in Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/experiments/pyterrier_kwiksort_learned.ipynb).

## Setup

Follow the [common setup](index.md#common-setup) to load the datasets, build the index, define the `bm25` baseline, and connect `ir_axioms` to the index.

## Define the Axioms

We list the axioms that serve as features for estimating the ORACLE.
Because some axioms call external APIs or are computationally expensive, we cache each axiom on disk:

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
    "ORIG": ORIG(),
}
axioms = [axiom.cached(cache_dir / "axioms" / name) for name, axiom in axioms.items()]
```

Including [ORIG](../axioms/utility.md#orig) as a feature lets the classifier learn when to trust the original ranking.

## Train the Estimator

We combine the preferences of all axioms in a random forest classifier.
The `EstimatorKwikSortReranker` is a PyTerrier `Estimator`: it is trained to predict the preferences of a target axiom (by default, ORACLE) and then re-ranks with KwikSort based on the predicted preferences:

```python
from sklearn.ensemble import RandomForestClassifier
from ir_axioms.integrations import EstimatorKwikSortReranker
from ir_axioms.tools import MiddlePivotSelection

random_forest = RandomForestClassifier(max_depth=3)

kwiksort_random_forest = bm25 % 20 >> EstimatorKwikSortReranker(
    axioms=axioms,
    estimator=random_forest,
    pivot_selection=MiddlePivotSelection(),
    verbose=True,
)
```

After setting up the trainable pipeline, we pass in training topics and relevance judgments:

```python
kwiksort_random_forest.fit(dataset_train.get_topics(), dataset_train.get_qrels())
```

## Experimental Evaluation

We compare the re-ranked pipeline to the BM25 baseline on the TREC 2020 topics, appending the baseline results after the top-20 positions using the `^` operator:

```python
from ir_measures import nDCG, MAP, RR
from pyterrier import Experiment

experiment = Experiment(
    [bm25, kwiksort_random_forest ^ bm25],
    dataset_test.get_topics(),
    dataset_test.get_qrels(),
    [nDCG @ 10, RR, MAP],
    names=["BM25", "KwikSort Random Forest"],
    verbose=True,
)
experiment.sort_values(by="nDCG@10", ascending=False)
```

## Feature Importances

Inspecting the feature importances of the random forest helps to identify axioms that are not used for re-ranking.
If an axiom's feature importance is zero for most of your applications, consider omitting it from the pipeline to save computation time:

```python
random_forest.feature_importances_
```
