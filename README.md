[![CI status](https://img.shields.io/github/actions/workflow/status/webis-de/ir_axioms/ci.yml?branch=main&style=flat-square)](https://github.com/webis-de/ir_axioms/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/webis-de/ir_axioms?style=flat-square)](https://codecov.io/github/webis-de/ir_axioms/)
[![Maintenance](https://img.shields.io/maintenance/yes/2025?style=flat-square)](https://github.com/webis-de/ir_axioms/graphs/contributors)  
[![PyPI version](https://img.shields.io/pypi/v/ir-axioms?style=flat-square)](https://pypi.org/project/ir-axioms/)
[![PyPI downloads](https://img.shields.io/pypi/dm/ir-axioms?style=flat-square)](https://pypi.org/project/ir-axioms/)
[![Python versions](https://img.shields.io/pypi/pyversions/ir-axioms?style=flat-square)](https://pypi.org/project/ir-axioms/)  
[![Issues](https://img.shields.io/github/issues/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/pulls)
[![Commit activity](https://img.shields.io/github/commit-activity/m/webis-de/ir_axioms?style=flat-square)](https://github.com/webis-de/ir_axioms/commits)
[![License](https://img.shields.io/github/license/webis-de/ir_axioms?style=flat-square)](LICENSE)

# ↕️ ir_axioms

Axiomatic constraints for information retrieval and retrieval-augmented generation.

_Note: If you're looking out for `ir_axioms`<1.0, please [go here]()._

## Installation

1. Install [Python 3.11](https://python.org/downloads/) or later.
2. Create and activate a virtual environment:

    ```shell
    python3 -m venv venv/
    source venv/bin/activate
    ```

3. Install project dependencies:

    ```shell
    pip install -e .
    ```

## Usage

Run the CLI with:

```shell
axioms --help
```

TODO: Explain commands to replicate experiments.

## Development

Refer to the general [installation instructions](#installation) to set up the development environment and install the dependencies.
Then, also install the test dependencies:

```shell
pip install -e .[tests]
```

After having implemented a new feature, please check the code format, inspect common LINT errors, and run all unit tests with the following commands:

```shell
ruff check .                   # Code format and LINT
mypy .                         # Static typing
bandit -c pyproject.toml -r .  # Security
pytest .                       # Unit tests
```

## Contribute

If you have found an important feature missing from our tool, please suggest it by creating an [issue](https://github.com/webis-de/ir_axioms/issues). We also gratefully accept [pull requests](https://github.com/webis-de/ir_axioms/pulls)!

If you are unsure about anything, post an [issue](https://github.com/webis-de/ir_axioms/issues/new) or contact us:

- [heinrich.merker@uni-jena.de](mailto:heinrich.merker@uni-jena.de)

We are happy to help!

## License

This repository is released under the [MIT license](LICENSE).
Files in the `data/` directory are exempt from this license.
