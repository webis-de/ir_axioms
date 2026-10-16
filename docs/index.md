---
hide:
  - navigation
---

<div class="hero" markdown>

# ↕️ ir_axioms

<p class="subtitle">Axiomatic constraints for information retrieval and retrieval-augmented generation</p>

`ir_axioms` is a Python framework for experimenting with axioms in information retrieval in a declarative way.
It includes reference implementations of many commonly used traditional retrieval axioms as well as axioms for retrieval-augmented generation (RAG).

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[Browse the axioms](axioms/index.md){ .md-button }

</div>

The library is well integrated with the [PyTerrier](https://github.com/terrier-org/pyterrier) framework and supports [Pyserini](https://github.com/castorini/pyserini) retrieval indices too.
Use-cases of `ir_axioms` include search-result re-ranking, analyses of (neural) retrieval systems, and analyses of generated RAG answers.

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

- :material-robot-outline: **RAG axioms**

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

## Publications

Read more about the `ir_axioms` framework and the new RAG axioms in these publications:

- Alexander Bondarenko, Maik Fröbe, Jan Heinrich Reimer, Benno Stein, Michael Völske, and Matthias Hagen.
  **[Axiomatic Retrieval Experimentation with `ir_axioms`](https://webis.de/publications.html?q=axiom#bondarenko_2022d).**
  SIGIR 2022. [@bondarenko2022axiomatic]
- Jan Heinrich Merker, Maik Fröbe, Benno Stein, Martin Potthast, and Matthias Hagen.
  **[Axioms for Retrieval-Augmented Generation](https://webis.de/publications.html?q=axiom#merker_2025b).**
  ICTIR 2025. [@merker2025axioms]

If you use `ir_axioms` in your research, please cite the papers above.
The BibTeX entries are listed in the [references](#references) below, and all references used in this documentation are available in [`references.bib`](references.bib).

!!! note "Looking for `ir_axioms` < 1.0?"

    The legacy version is still available on the [`legacy` branch](https://github.com/webis-de/ir_axioms/tree/legacy).
    To upgrade your code, follow the [migration guide](getting-started/migration.md).
