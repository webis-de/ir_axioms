# Walkthroughs

<p class="subtitle">Step-by-step tutorials based on our example notebooks</p>

These walkthroughs demonstrate the main use cases of `ir_axioms` on standard benchmark data.
They build on the [integrations](../getting-started/integrations.md) and use axioms from the [axiom catalog](../axioms/index.md).
Each walkthrough is based on one of the [Jupyter notebooks](https://github.com/webis-de/ir_axioms/tree/main/experiments) in the repository.

| Walkthrough | You will learn to… | Notebook |
|:--|:--|:--|
| [Hands-on tutorial](hands-on.md) | analyze retrieval systems axiomatically, find and fix ranking errors by axiomatic re-ranking | [`grenoble2025_showcase.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/grenoble2025_showcase.ipynb) |
| [KwikSort re-ranking](kwiksort.md) | re-rank BM25 results with a vote of many axioms | [`pyterrier_kwiksort.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_kwiksort.ipynb) |
| [Learning the ORACLE](kwiksort-learned.md) | train a classifier that combines axioms to predict relevance preferences | [`pyterrier_kwiksort_learned.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_kwiksort_learned.ipynb) |
| [Axiomatic LTR features](ltr-features.md) | use aggregated axiom preferences as features for LambdaMART | [`pyterrier_ltr_features.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_ltr_features.ipynb) |
| [Post-hoc analysis](post-hoc-analysis.md) | analyze the consistency of runs and qrels with axioms | [`pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb) |
| [Parallel preferences](parallel-preferences.md) | compute preferences for many TREC runs in parallel | [`pyterrier_preferences_parallel.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/pyterrier_preferences_parallel.ipynb) |
| [Evaluating RAG responses](rag-evaluation.md) | compare RAG axioms to human and nugget-based judgments | [`ictir2025.ipynb`](https://github.com/webis-de/ir_axioms/blob/main/experiments/ictir2025.ipynb) |

!!! note "API versions"

    The walkthroughs use the current `ir_axioms` ≥ 1.0 API.
    Some of the linked notebooks were written for the legacy API (e.g., using `~axiom` for caching or `ir_axioms.backend.pyterrier` imports).
    See the [migration guide](../getting-started/migration.md) for the differences.

## Common Setup

The retrieval walkthroughs use the passage ranking task of the TREC Deep Learning tracks 2019 and 2020 on MS MARCO, retrieved with PyTerrier.

### Install Dependencies

```shell
pip install "ir_axioms[pyterrier,experiments]"
python -m spacy download en_core_web_sm
```

### Load Datasets

We load the datasets with PyTerrier's `get_dataset()`, which uses [`ir_datasets`](https://ir-datasets.com/) under the hood:

```python
from pyterrier.datasets import get_dataset

dataset_name = "msmarco-passage"
dataset = get_dataset(f"irds:{dataset_name}")
dataset_train = get_dataset(f"irds:{dataset_name}/trec-dl-2019/judged")
dataset_test = get_dataset(f"irds:{dataset_name}/trec-dl-2020/judged")
```

### Build the Index

If the index does not exist yet, index all MS MARCO passages (this takes a while):

```python
from pathlib import Path
from pyterrier.terrier import IterDictIndexer

cache_dir = Path("cache/")
index_dir = cache_dir / "indices" / dataset_name

if not index_dir.exists():
    indexer = IterDictIndexer(str(index_dir.absolute()))
    indexer.index(dataset.get_corpus_iter(), fields=["text"])
```

### Baseline Retrieval

We use BM25 as the baseline retrieval model:

```python
from pyterrier.terrier import Retriever

bm25 = Retriever(str(index_dir.absolute()), wmodel="BM25")
```

### Connect `ir_axioms` to the Index

Finally, let `ir_axioms` use the index for tokenization and collection statistics, and `ir_datasets` for the query and document texts:

```python
from ir_axioms.integrations import inject_pyterrier

inject_pyterrier(
    index_location=index_dir,
    dataset=dataset_name,
)
```

Call `inject_pyterrier()` **before** instantiating any axioms, as the axioms' dependencies are resolved when they are created.
