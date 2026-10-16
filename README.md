[![CI status](https://img.shields.io/github/actions/workflow/status/webis-de/ir_axioms/ci.yml?branch=main&style=flat-square)](https://github.com/webis-de/ir_axioms/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/webis-de/ir_axioms?style=flat-square)](https://codecov.io/github/webis-de/ir_axioms/)
[![Documentation](https://img.shields.io/github/actions/workflow/status/webis-de/ir_axioms/docs.yml?branch=main&style=flat-square&label=docs)](https://ir_axioms.webis.de/)
[![Maintenance](https://img.shields.io/maintenance/yes/2026?style=flat-square)](https://github.com/webis-de/ir_axioms/graphs/contributors)  
[![PyPI version](https://img.shields.io/pypi/v/ir-axioms?style=flat-square)](https://pypi.org/project/ir-axioms/)
[![PyPI downloads](https://img.shields.io/pypi/dm/ir-axioms?style=flat-square)](https://pypi.org/project/ir-axioms/)
[![Python versions](https://img.shields.io/pypi/pyversions/ir-axioms?style=flat-square)](https://pypi.org/project/ir-axioms/)  
[![Issues](https://img.shields.io/github/issues/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/pulls)
[![Commit activity](https://img.shields.io/github/commit-activity/m/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/commits)
[![License](https://img.shields.io/github/license/webis-de/ir_axioms?style=flat-square)](LICENSE)

# ↕️ ir_axioms

Axiomatic constraints for information retrieval and retrieval-augmented generation.

`ir_axioms` is a Python framework for experimenting with axioms in information retrieval in a declarative way.
It includes reference implementations of [many commonly used traditional retrieval axioms](https://ir_axioms.webis.de/axioms/) as well as [axioms for retrieval-augmented generation](https://ir_axioms.webis.de/axioms/generation/).
The library is well integrated with the [PyTerrier](https://github.com/terrier-org/pyterrier) framework and supports [Pyserini](https://github.com/castorini/pyserini) retrieval indices too.
Use-cases of `ir_axioms` include [search-result re-ranking](https://ir_axioms.webis.de/walkthroughs/kwiksort/), [analyses of (neural) retrieval systems](https://ir_axioms.webis.de/walkthroughs/post-hoc-analysis/), and [analyses of generated RAG answers](https://ir_axioms.webis.de/walkthroughs/rag-evaluation/).

Refer to the `ir_axioms` [documentation](https://ir_axioms.webis.de/) for:

- [Installation](https://ir_axioms.webis.de/getting-started/installation/) and a [quickstart](https://ir_axioms.webis.de/getting-started/quickstart/)
- [Axiom intuitions, formalizations, and implementations](https://ir_axioms.webis.de/axioms/)
- [Related information retrieval literature](https://ir_axioms.webis.de/axioms/related/)
- [Walkthroughs](https://ir_axioms.webis.de/walkthroughs/) on re-ranking, post-hoc analyses, and RAG evaluation
- [Instructions for developing new axioms](https://ir_axioms.webis.de/development/axiom-development/)

## TL;DR

To get started without reading the [documentation](https://ir_axioms.webis.de/), follow these steps:

1. Install the package from PyPI:

   ```shell
   pip install ir_axioms>=1.0
   ```

2. Install a spaCy model:

   ```shell
   python -m spacy download en_core_web_sm
   ```

3. Load and combine axioms:

   ```python
   from ir_axioms.axiom import TFC1, PROX1, ORIG

   axiom = (TFC1() & PROX1()) | ORIG()
   ```

4. Define some test data:

   ```python
   from ir_axioms.model import Query, Document

   query = Query(id="q1", text="axiomatic information retrieval")
   documents = [
      Document(id="d1", text="Retrieval axioms formalize constraints for retrieval models.", rank=1),
      Document(id="d2", text="Axiomatic information retrieval: axiomatic constraints guide retrieval.", rank=2),
   ]
   ```

5. Compute axiom preferences or re-rank documents:

   ```python
   axiom.preferences(input=query, outputs=documents)
   axiom.rerank_kwiksort(input=query, ranking=documents)
   ```

## Contribute

If you have found an important feature missing from our tool, please suggest it by creating an [issue](https://github.com/webis-de/ir_axioms/issues). We also gratefully accept [pull requests](https://github.com/webis-de/ir_axioms/pulls)!
Get started by following our [development guides](https://ir_axioms.webis.de/development/).

## License

This repository is released under the [MIT license](LICENSE).
Files in the `data/` directory are exempt from this license.
If you use `ir_axioms` in your research, consider [citing us](https://ir_axioms.webis.de/#references).
