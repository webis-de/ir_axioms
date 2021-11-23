# ↕️ ir_axioms

`ir_axioms` is a python package that provides a common, intuitive interface to many IR axioms.
The package helps you to rerank and evaluate runs from various information retrieval frameworks 
such as [Pyserini](https://github.com/castorini/pyserini).

## Usage

### Backends

## Development

### Installation

Install dependencies for developing the `ir_axioms` package:
```shell
pip install -e .
```

If you want to develop the Pyserini backend, install dependencies like this:
```shell
pip install -e .[pyserini]
```

### Build wheel

```shell
python -m build
```