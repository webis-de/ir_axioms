 [![CI/CD status](https://git.webis.de/code-research/web-search/ir_axioms/badges/main/pipeline.svg?style=flat-square)](https://git.webis.de/code-research/web-search/ir_axioms/-/pipelines?ref=main)
[![Test coverage](https://git.webis.de/code-research/web-search/ir_axioms/badges/main/coverage.svg?style=flat-square)](https://git.webis.de/code-research/web-search/ir_axioms/-/pipelines?ref=main)

# ↕️ ir_axioms

`ir_axioms` is a python package that provides a common, intuitive interface to many IR axioms.
The package helps you to rerank and evaluate runs from various information retrieval frameworks 
such as [Pyserini](https://github.com/castorini/pyserini).

## Usage

### Backends

### Example Notebooks
We include several example notebooks which demonstrate reranking and evaluating preferences in [PyTerrier](https://github.com/terrier-org/pyterrier).
You can find the examples in the [`examples/` directory](examples/).

[![Open in Colab](https://img.shields.io/badge/notebook-open%20in%20colab-informational)](https://colab.research.google.com/github/webis-de/ir_axioms/blob/main/examples/pyterrier.ipynb)

## Development

To build and develop this package you need to install the `build`, and `setuptools` and `wheel` packages:
```shell
pip install build setuptools wheel
```
(On most systems, these packages are already installed.)

### Installation

Install dependencies for developing the `ir_axioms` package:
```shell
pip install -e .
```

If you want to develop the [Pyserini](https://github.com/castorini/pyserini) backend, install dependencies like this:
```shell
pip install -e .[pyserini]
```

If you want to develop the [PyTerrier](https://github.com/terrier-org/pyterrier) backend, install dependencies like this:
```shell
pip install -e .[pyterrier]
```

### Testing

Verify your changes against our test suite to verify.
```shell
flake8
pylint -E ir_axioms
pytest
```

Please also add tests for the axioms or integrations you've added.

### Build wheel

A wheel for this package can be built by:
```shell
python -m build
```
