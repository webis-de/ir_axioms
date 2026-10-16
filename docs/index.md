---
hide:
  - navigation
---

<div class="hero" markdown>

# ir_axioms

<p class="subtitle">Axiomatic constraints for information retrieval and retrieval-augmented generation</p>

`ir_axioms` is a Python framework for experimenting with axioms in information retrieval in a declarative way.
It includes reference implementations of many commonly used traditional retrieval axioms as well as axioms for retrieval-augmented generation (RAG) [@bondarenko2022axiomatic].

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[Browse axioms](axioms/index.md){ .md-button }
[View on GitHub](https://github.com/webis-de/ir_axioms){ .md-button }
[View on PyPI](https://pypi.org/project/ir-axioms/){ .md-button }

</div>

The library is well [integrated](getting-started/integrations.md) with the [PyTerrier](https://github.com/terrier-org/pyterrier) framework and supports [Pyserini](https://github.com/castorini/pyserini) retrieval indices too.
Use-cases of `ir_axioms` include [search-result re-ranking](walkthroughs/kwiksort.md), [analyses of (neural) retrieval systems](walkthroughs/post-hoc-analysis.md), and [analyses of generated RAG answers](walkthroughs/rag-evaluation.md).

```python
from ir_axioms.axiom import TFC1, PROX1, ORIG
from ir_axioms.model import Query, Document

query = Query(id="q1", text="axiomatic information retrieval")
documents = [
    Document(id="d1", text="Retrieval axioms formalize constraints for retrieval models.", rank=1),
    Document(id="d2", text="Axiomatic information retrieval: axiomatic constraints guide retrieval.", rank=2),
]

axiom = (TFC1() & PROX1()) | ORIG()  # (1)!
axiom.preferences(input=query, outputs=documents)  # (2)!
axiom.rerank_kwiksort(input=query, ranking=documents)  # (3)!
```

1. Combine axioms declaratively: TFC1 and PROX1 must agree, otherwise fall back to the original ranking.
2. Compute the pairwise preference matrix for all documents.
3. Re-rank the documents with KwikSort based on the axiom's preferences.

## Features

<div class="grid cards" markdown>

- :material-scale-balance: **Retrieval axioms**

    ---

    Reference implementations of 30+ retrieval axioms, from TFC1 to PROX5, with documented deviations from their original formulations.

    [:octicons-arrow-right-24: Retrieval axioms](axioms/retrieval.md)

- :material-robot-outline: **RAG axioms**  [@merker2025axioms]

    ---

    Axioms for retrieval-augmented generation that address coherence, coverage, consistency, correctness, and clarity of generated responses.

    [:octicons-arrow-right-24: RAG axioms](axioms/generation.md)

- :material-pipe: **PyTerrier integration**

    ---

    Re-rank with KwikSort, compute preference features for learning to rank, and run axiomatic experiments directly in PyTerrier pipelines.

    [:octicons-arrow-right-24: Integrations](getting-started/integrations.md)

- :material-school-outline: **Walkthroughs**

    ---

    Step-by-step tutorials on re-ranking, post-hoc analyses of runs and qrels, and evaluating RAG responses.

    [:octicons-arrow-right-24: Walkthroughs](walkthroughs/index.md)

</div>

When using `ir_axioms` in your research, consider citing our SIGIR 2022 paper about the framework [@bondarenko2022axiomatic] or our ICTIR 2025 paper about the RAG axioms [@merker2025axioms].
