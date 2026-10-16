# KwikSort Re-Ranking

<p class="subtitle">Re-rank BM25 results with a majority vote of many axioms</p>

This walkthrough shows how `ir_axioms` can re-rank a result set in PyTerrier using the KwikSort algorithm [@ailon2008aggregating] and manually combined axioms [@hagen2016axiomatic].
We use the passage ranking task of the TREC 2020 Deep Learning track with BM25 as the baseline and evaluate nDCG@10, reciprocal rank, and average precision with PyTerrier's standard `Experiment` functionality.

Based on the notebook [`pyterrier_kwiksort.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_kwiksort.ipynb)
[![Open in Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/experiments/pyterrier_kwiksort.ipynb).

## Setup

Follow the [common setup](index.md#common-setup) to load the datasets, build the index, define the `bm25` baseline, and connect `ir_axioms` to the index.

## Combine Axioms

We combine many of the axioms implemented in `ir_axioms` in a majority vote.
That is, we only keep preferences where at least 50 % of the axioms agree, and fall back to the original BM25 ranking otherwise.
Because some axioms call external APIs (e.g., the argumentativeness axioms) or are computationally expensive, we cache the voting result on disk:

```python
from ir_axioms.axiom import (
    ArgUC, QTArg, QTPArg, aSL,
    LNC1, TF_LNC, LB1,
    PROX1, PROX2, PROX3, PROX4, PROX5,
    REG, ANTI_REG, ASPECT_REG,
    AND, LEN_AND, M_AND, LEN_M_AND, DIV, LEN_DIV,
    TFC1, TFC3, M_TDC, LEN_M_TDC,
    STMC1, STMC2,
    ORIG, MajorityVoteAxiom,
)

axiom = MajorityVoteAxiom(
    axioms=[
        ArgUC(), QTArg(), QTPArg(), aSL(),
        LNC1(), TF_LNC(), LB1(),
        PROX1(), PROX2(), PROX3(), PROX4(), PROX5(),
        REG(), ANTI_REG(), ASPECT_REG(),
        AND(), LEN_AND(), M_AND(), LEN_M_AND(), DIV(), LEN_DIV(),
        TFC1(), TFC3(), M_TDC(), LEN_M_TDC(),
        STMC1(), STMC2(),
    ],
    minimum_votes=0.5,
).cached(cache_dir / "axioms" / "vote") | ORIG()
```

## KwikSort Re-Ranking

With the combined axiom, we create a new PyTerrier pipeline that re-ranks the top-20 baseline results using KwikSort.
KwikSort works similar to Quicksort, but instead of comparing items by their natural order, it compares them by axiom preferences.
Here, we always choose the middle document as the pivot, to make the re-ranking deterministic:

```python
from ir_axioms.integrations import KwikSortReranker
from ir_axioms.tools import MiddlePivotSelection

kwiksort = bm25 % 20 >> KwikSortReranker(
    axiom=axiom,
    pivot_selection=MiddlePivotSelection(),
    verbose=True,
)
```

## Experimental Evaluation

Because the axiomatic re-ranker is a PyTerrier transformer, we can use PyTerrier's `Experiment` to compare the re-ranked pipeline to the BM25 baseline.
Refer to the [PyTerrier documentation](https://pyterrier.readthedocs.io/en/latest/experiments.html) to learn more about running experiments.
We append the baseline results after the top-20 positions using the `^` operator:

```python
from ir_measures import nDCG, MAP, RR
from pyterrier import Experiment

experiment = Experiment(
    [bm25, kwiksort ^ bm25],
    dataset_test.get_topics(),
    dataset_test.get_qrels(),
    [nDCG @ 10, RR, MAP],
    names=["BM25", "KwikSort"],
    verbose=True,
)
experiment.sort_values(by="nDCG@10", ascending=False)
```

## Next Steps

Instead of weighting all axioms equally, [learn how to combine them](kwiksort-learned.md) from relevance judgments.
