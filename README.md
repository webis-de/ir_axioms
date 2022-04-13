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
It includes reference implementations of many commonly used retrieval axioms and is well integrated with the [PyTerrier](https://github.com/terrier-org/pyterrier) and [Pyserini](https://github.com/castorini/pyserini) frameworks.
Re-rank your search results today with `ir_axioms` and understand your retrieval systems better by analyzing
axiomatic preferences!

## Usage

### Example Notebooks

We include several example notebooks to demonstrate re-ranking and preference evaluation in [PyTerrier](https://github.com/terrier-org/pyterrier) using `ir_axioms`.
You can find all examples in the [`examples/` directory](examples).

- [Re-ranking top-20 results using KwikSort](examples/pyterrier_kwiksort.ipynb)
  [![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/pyterrier_kwiksort.ipynb)
- [Re-ranking top-20 results using KwikSort learned from ORACLE preferences](examples/pyterrier_kwiksort_learned.ipynb)
  [![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/pyterrier_kwiksort_learned.ipynb)
- [Re-ranking top-20 results using LambdaMART with axiomatic preference features](examples/pyterrier_ltr_features.ipynb)
  [![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/pyterrier_ltr_features.ipynb)
- [Post-Hoc Analysis of Rankings and
Relevance Judgments](examples/pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb)[![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/pyterrier_post_hoc_analysis_of_runs_and_qrels.ipynb)
- [Axiomatic prefernces for TREC Deep Learning 2019 runs (passages)](examples/trec_28_deep_passages_preferences_depth_10.ipynb)
  [![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/trec_28_deep_passages_preferences_depth_10.ipynb)
- [Axiomatic prefernces for TREC Deep Learning 2019 runs (documents)](examples/trec_28_deep_documents_preferences_depth_10.ipynb)
  [![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/trec_28_deep_documents_preferences_depth_10.ipynb)
- [Axiomatic prefernces for TREC Deep Learning 2020 runs (passages)](examples/trec_29_deep_passages_preferences_depth_10.ipynb)
  [![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/trec_29_deep_passages_preferences_depth_10.ipynb)
- [Axiomatic prefernces for TREC Deep Learning 2020 runs (documents)](examples/trec_29_deep_documents_preferences_depth_10.ipynb)
  [![Launch Binder](https://img.shields.io/badge/launch-binder-informational?style=flat-square)](https://mybinder.org/v2/gh/webis-de/ir_axioms/main?labpath=examples/trec_29_deep_documents_preferences_depth_10.ipynb)

### Backends

TODO

### Slurm

If you want to play around with `ir_axioms` in Jupyter Lab, you can use this command to provision a server via Slurm:

```shell
scripts/slurm-start-jupyter-lab.sh
```

## Citation

If you use this package or its components in your research, please cite the following paper describing the `ir_axioms`
framework and its use-cases:

> Alexander Bondarenko, Maik Fröbe, Jan Heinrich Reimer, Benno Stein, Michael Völske, and Matthias Hagen. [Axiomatic Retrieval Experimentation with `ir_axioms`](https://webis.de/publications.html?q=ir_axioms#bondarenko_2022d). In _45th International ACM Conference on Research and Development in Information Retrieval (SIGIR 2022)_, July 2022. ACM.

You can use the following BibTeX entry for citation:

```bibtex
@InProceedings{bondarenko:2022d,
  author =                {Alexander Bondarenko and Maik Fr{\"o}be and {Jan Heinrich} Reimer and Benno Stein and Michael V{\"o}lske and Matthias Hagen},
  booktitle =             {45th International ACM Conference on Research and Development in Information Retrieval (SIGIR 2022)},
  month =                 jul,
  publisher =             {ACM},
  site =                  {Madrid, Spain},
  title =                 {{Axiomatic Retrieval Experimentation with ir_axioms}},
  year =                  2022
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

## License

This repository is released under the [MIT license](LICENSE). If you use `ir_axioms` in your research, we'd be glad if
you'd [cite us](#citation).
