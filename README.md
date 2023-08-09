[![Paper DOI](https://img.shields.io/badge/DOI-10.1145%2F3477495.3531743-blue?style=flat-square)](https://doi.org/10.1145/3477495.3531743)
[![CI status](https://img.shields.io/github/actions/workflow/status/webis-de/ir_axioms/ci.yml?branch=main&style=flat-square)](https://github.com/webis-de/ir_axioms/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/webis-de/ir_axioms?style=flat-square)](https://codecov.io/github/webis-de/ir_axioms/)
[![Maintenance](https://img.shields.io/maintenance/yes/2023?style=flat-square)](https://github.com/webis-de/ir_axioms/graphs/contributors)  
[![PyPI version](https://img.shields.io/pypi/v/ir_axioms?style=flat-square)](https://pypi.org/project/ir_axioms/)
[![PyPI downloads](https://img.shields.io/pypi/dm/ir_axioms?style=flat-square)](https://pypi.org/project/ir_axioms/)
[![Python versions](https://img.shields.io/pypi/pyversions/ir_axioms?style=flat-square)](https://pypi.org/project/ir_axioms/)  
[![Docker version](https://img.shields.io/docker/v/webis/ir_axioms?style=flat-square&label=docker
)](https://hub.docker.com/repository/docker/webis/ir_axioms)
[![Docker pulls](https://img.shields.io/docker/pulls/webis/ir_axioms?style=flat-square&label=pulls)](https://hub.docker.com/repository/docker/webis/ir_axioms)
[![Docker image size](https://img.shields.io/docker/image-size/webis/ir_axioms?style=flat-square)](https://hub.docker.com/repository/docker/webis/ir_axioms)  
[![Issues](https://img.shields.io/github/issues/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/pulls)
[![Commit activity](https://img.shields.io/github/commit-activity/m/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/commits)
[![License](https://img.shields.io/github/license/webis-de/ir_axioms?style=flat-square)](LICENSE)

# ↕️ ir_axioms

Intuitive axiomatic retrieval experimentation.

`ir_axioms` is a Python framework for experimenting with axioms in information retrieval in a declarative way. 
It includes reference implementations of many commonly used retrieval axioms and is well integrated with the [PyTerrier](https://github.com/terrier-org/pyterrier) framework and the [Pyserini](https://github.com/castorini/pyserini) toolkit.
Re-rank your search results today with `ir_axioms` and understand your retrieval systems better by analyzing
axiomatic preferences!

|            [Presentation video on YouTube](https://youtu.be/hZtWw805TBg)             |                  [Poster](https://webis.de/downloads/publications/posters/bondarenko_2022d.pdf)                   |
|:------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------:|
| [![Presentation video](documentation/video-cover.png)](https://youtu.be/hZtWw805TBg) | [![Poster](documentation/poster-cover.png)](https://webis.de/downloads/publications/posters/bondarenko_2022d.pdf) |

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

| Transformer Class           | Type                      | Description                                                  |
|:----------------------------|:--------------------------|:-------------------------------------------------------------|
| `AggregatedPreferences`     | 𝑅 → 𝑅<sub>𝑓</sub>      | Aggregate axiom preferences for each document                |
| `EstimatorKwikSortReranker` | 𝑅 → 𝑅′                  | Train estimator for ORACLE, use it to re-rank with KwikSort. |
| `KwikSortReranker`          | 𝑅 → 𝑅′                  | Re-rank using axiom preferences aggregated by KwikSort.      |
| `PreferenceMatrix`          | 𝑅 → (𝑅×𝑅)<sub>𝑓</sub> | Compute an axiom’s preference matrix.                        |

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

### TIRA

Here's an example how `ir_axioms` can be used to get axiomatic preferences for a run in TIRA:

```shell
tira-run \
  --input-directory data/tira/input \
  --output-directory data/tira/output \
  --image webis/ir_axioms \
  --command '/venv/bin/python -m ir_axioms \
    --terrier-version 5.7 \
    --terrier-helper-version 0.0.7 \
    --offline \
    preferences \
    --run-file $inputDataset/run.jsonl \
    --run-format jsonl \
    --index-dir $inputDataset/index \
    --output-dir $outputDir \
    TFC1'
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

Install package and test dependencies:

```shell
pip install -e .[test]
```

### Testing

Install test dependencies:

```shell
pip install -e .[test]
```

### Testing

Verify your changes against our test suite to verify.

```shell
flake8 ir_axioms tests examples
pylint -E ir_axioms tests examples
pytest ir_axioms tests examples
```

Please also add tests for your newly developed code.

### Build wheels

Wheels for this package can be built with:

```shell
python -m build
```

## Support

If you hit any problems using `ir_axioms` or reproducing our experiments, please write us an email or file an [issue](https://github.com/webis-de/ir_axioms/issues/new):

- [heinrich.reimer@uni-jena.de](mailto:heinrich.reimer@uni-jena.de)
- [maik.froebe@uni-jena.de](mailto:maik.froebe@uni-jena.de)
- [alexander.bondarenko@uni-jena.de](mailto:alexander.bondarenko@uni-jena.de)

We're happy to help!

## License

This repository is released under the [MIT license](LICENSE). If you use `ir_axioms` in your research, we'd be glad if
you'd [cite us](#citation).

## Abstract
Axiomatic approaches to information retrieval have played a key role in determining basic constraints that characterize good retrieval models. Beyond their importance in retrieval theory, axioms have been operationalized to improve an initial ranking, to “guide” retrieval, or to explain some model’s rankings. However, recent open-source retrieval frameworks like PyTerrier and Pyserini, which made it easy to experiment with sparse and dense retrieval models, have not included any retrieval axiom support so far. To fill this gap, we propose `ir_axioms`, an open-source Python framework that integrates retrieval axioms with common retrieval frameworks. We include reference implementations for 25 retrieval axioms, as well as components for preference aggregation, re-ranking, and evaluation. New axioms can easily be defined by implementing an abstract data type or by intuitively combining existing axioms with Python operators or regression. Integration with PyTerrier and `ir_datasets` makes standard retrieval models, corpora, topics, and relevance judgments—including those used at TREC—immediately accessible for axiomatic experimentation. Our experiments on the TREC Deep Learning tracks showcase some potential research questions that ir_axioms can help to address.
