[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3477495.3531743-blue?style=flat-square)](https://doi.org/10.1145/3477495.3531743)
[![PyPi](https://img.shields.io/pypi/v/ir_axioms?style=flat-square)](https://pypi.org/project/ir_axioms/)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/webis-de/ir_axioms.svg?style=flat-square)](https://lgtm.com/projects/g/webis-de/ir_axioms)
[![CI](https://img.shields.io/github/workflow/status/webis-de/ir_axioms/CI?style=flat-square)](https://github.com/webis-de/ir_axioms/actions?query=workflow%3A"CI")
[![Code coverage](https://img.shields.io/codecov/c/github/webis-de/ir_axioms?style=flat-square)](https://codecov.io/github/webis-de/ir_axioms/)
[![Python](https://img.shields.io/pypi/pyversions/ir_axioms?style=flat-square)](https://pypi.org/project/ir_axioms/)
[![Issues](https://img.shields.io/github/issues/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/commits)
[![Downloads](https://img.shields.io/pypi/dm/ir_axioms?style=flat-square)](https://pypi.org/project/ir_axioms/)
[![License](https://img.shields.io/github/license/webis-de/ir_axioms?style=flat-square)](LICENSE)

# ↕️ ir_axioms

Intuitive axiomatic retrieval experimentation.

`ir_axioms` is a Python framework for experimenting with axioms in information retrieval in a declarative way. 
It includes reference implementations of many commonly used retrieval axioms and is well integrated with the [PyTerrier](https://github.com/terrier-org/pyterrier) framework and the [Pyserini](https://github.com/castorini/pyserini) toolkit.
Re-rank your search results today with `ir_axioms` and understand your retrieval systems better by analyzing
axiomatic preferences!

|            [Presentation video on YouTube](https://youtu.be/hZtWw805TBg)             |
|:------------------------------------------------------------------------------------:|
| [![Presentation video](documentation/video-cover.png)](https://youtu.be/hZtWw805TBg) |

## Usage

The `ir_axioms` framework is easy to use. Below, we've prepared some notebooks showcasing the main features.
If you have questions or need assistance, please [contatct us](#support).

### Example Notebooks

We include several example notebooks to demonstrate re-ranking and preference evaluation in [PyTerrier](https://github.com/terrier-org/pyterrier) using `ir_axioms`.
You can find all examples in the [`examples/` directory](examples).

- [Re-ranking top-20 results using KwikSort](examples/pyterrier_kwiksort.ipynb)
  [![Launch in Google Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/examples/pyterrier_kwiksort.ipynb)
- [Re-ranking top-20 results using KwikSort learned from ORACLE preferences](examples/pyterrier_kwiksort_learned.ipynb)
  [![Launch in Google Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/examples/pyterrier_kwiksort_learned.ipynb)
- [Re-ranking top-20 results using LambdaMART with axiomatic preference features](examples/pyterrier_ltr_features.ipynb)
  [![Launch in Google Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/examples/pyterrier_ltr_features.ipynb)
- [Post-Hoc analysis of rankings and relevance judgments](examples/pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb)
  [![Launch in Google Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/examples/pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb)
- [Computing axiom preferences for top-20 results of TREC 2022 Deep Learning (passage) runs in parallel](examples/pyterrier_preferences_parallel.ipynb)
  [![Launch in Google Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/examples/pyterrier_preferences_parallel.ipynb)
- [SIGIR 2022 showcase](examples/sigir2022_showcase.ipynb) for step-by-step explanations with our [presentation video](https://youtu.be/hZtWw805TBg)
  [![Launch in Google Colab](https://img.shields.io/badge/open%20in-colab-informational?style=flat-square)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/examples/sigir2022_showcase.ipynb)

### Backends

You can experiment with `ir_axioms` in PyTerrier and Pyserini.
However, we recommend PyTerrier as not all features are implemented for the Pyserini backend.

#### PyTerrier (Terrier index)

To use `ir_axioms` with a Terrier index, please use our PyTerrier transformers (modules):
| Transformer Class           | Type       | Description                                                  |
|:----------------------------|:-----------|:-------------------------------------------------------------|
| `AggregatedPreferences`     | 𝑅 → 𝑅𝑓     | Aggregate axiom preferences for each document                |
| `EstimatorKwikSortReranker` | 𝑅 → 𝑅′     | Train estimator for ORACLE, use it to re-rank with KwikSort. |
| `KwikSortReranker`          | 𝑅 → 𝑅′     | Re-rank using axiom preferences aggregated by KwikSort.      |
| `PreferenceMatrix`          | 𝑅 → (𝑅×𝑅)𝑓 | Compute an axiom’s preference matrix.                        |

You can also directly instantiate a index context object from a Terrier index if you want to build custom axiomatic modules:

```python
from ir_axioms.backend.pyterrier import TerrierIndexContext
context = TerrierIndexContext("/path/to/index/dir")
axiom.preference(context, query, doc1, doc2)
```

#### Pyserini (Anserini index)

We don't have modules for Pyserini to re-rank or analyze results out of the box.
However, you can still comute axiom preferences to integrate retrieval axioms into your search pipeline:

```python
from ir_axioms.backend.pyserini import AnseriniIndexContext
context = AnseriniIndexContext("/path/to/index/dir")
axiom.preference(context, query, doc1, doc2)
```

## Citation

If you use this package or its components in your research, please cite the following paper describing the `ir_axioms`
framework and its use-cases:

> Alexander Bondarenko, Maik Fröbe, Jan Heinrich Reimer, Benno Stein, Michael Völske, and Matthias Hagen. [Axiomatic Retrieval Experimentation with `ir_axioms`](https://webis.de/publications.html?q=ir_axioms#bondarenko_2022d). In _45th International ACM Conference on Research and Development in Information Retrieval (SIGIR 2022)_, July 2022. ACM.

You can use the following BibTeX entry for citation:

```bibtex
@InProceedings{bondarenko:2022d,
  author =    {Alexander Bondarenko and
               Maik Fr{\"o}be and
               {Jan Heinrich} Reimer and
               Benno Stein and
               Michael V{\"o}lske and
               Matthias Hagen},
  booktitle = {45th International ACM Conference on Research and Development
               in Information Retrieval (SIGIR 2022)},
  month =     jul,
  publisher = {ACM},
  site =      {Madrid, Spain},
  title =     {{Axiomatic Retrieval Experimentation with ir_axioms}},
  year =      2022
}
```

## Development

To build `ir_axioms` and contribute to its development you need to install the `build`, and `setuptools` and `wheel` packages:

```shell
pip install build setuptools wheel
```

(On most systems, these packages are already pre-installed.)

### Installation

Install dependencies for developing the `ir_axioms` package:

```shell
pip install -e .
```

If you want to develop the [Pyserini](https://github.com/castorini/pyserini) backend, install dependencies like this:

```shell
pip install -e .[pyserini]
```

If you want to develop the [PyTerrier](https://github.com/terrier-org/pyterrier) backend, install dependencies like
this:

```shell
pip install -e .[pyterrier]
```

### Testing

Install test dependencies:

```shell
pip install -e .[test]
```

Verify your changes against our test suite to verify.

```shell
flake8 ir_axioms tests
pylint -E ir_axioms tests.unit --ignore-paths=^ir_axioms.backend
pytest ir_axioms/ tests/unit/ --ignore=ir_axioms/backend/
```

Please also add tests for the axioms or integrations you've added.

#### Testing backend integrations

Install test dependencies (replace `<BACKEND>` with either `pyserini` or `pyterrier`):

```shell
pip install -e .[<BACKEND>]
```

Verify your changes against our test suite to verify.

```shell
pylint -E ir_axioms.backend.<BACKEND> tests.integration.<BACKEND>
pytest tests/integration/<BACKEND>/
```

### Build wheel

A wheel for this package can be built by running:

```shell
python -m build
```

## Support

If you hit any problems using `ir_axioms` or reproducing our experiments, please write us an email or file an [issue](https://github.com/webis-de/ir_axioms/issues):

- [jan.reimer@student.uni-halle.de](mailto:jan.reimer@student.uni-halle.de)
- [maik.froebe@informatik.uni-halle.de](mailto:maik.froebe@informatik.uni-halle.de)
- [alexander.bondarenko@informatik.uni-halle.de](mailto:alexander.bondarenko@informatik.uni-halle.de)

We're happy to help!

## License

This repository is released under the [MIT license](LICENSE). If you use `ir_axioms` in your research, we'd be glad if
you'd [cite us](#citation).

## Abstract
Axiomatic approaches to information retrieval have played a key role in determining basic constraints that characterize good retrieval models. Beyond their importance in retrieval theory, axioms have been operationalized to improve an initial ranking, to “guide” retrieval, or to explain some model’s rankings. However, recent open-source retrieval frameworks like PyTerrier and Pyserini, which made it easy to experiment with sparse and dense retrieval models, have not included any retrieval axiom support so far. To fill this gap, we propose `ir_axioms`, an open-source Python framework that integrates retrieval axioms with common retrieval frameworks. We include reference implementations for 25 retrieval axioms, as well as components for preference aggregation, re-ranking, and evaluation. New axioms can easily be defined by implementing an abstract data type or by intuitively combining existing axioms with Python operators or regression. Integration with PyTerrier and `ir_datasets` makes standard retrieval models, corpora, topics, and relevance judgments—including those used at TREC—immediately accessible for axiomatic experimentation. Our experiments on the TREC Deep Learning tracks showcase some potential research questions that ir_axioms can help to address.
