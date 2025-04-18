[![CI status](https://img.shields.io/github/actions/workflow/status/janheinrichmerker/axioms/ci.yml?branch=main&style=flat-square)](https://github.com/janheinrichmerker/axioms/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/janheinrichmerker/axioms?style=flat-square)](https://codecov.io/github/janheinrichmerker/axioms/)
[![Maintenance](https://img.shields.io/maintenance/yes/2024?style=flat-square)](https://github.com/janheinrichmerker/axioms/graphs/contributors)  
[![PyPI version](https://img.shields.io/pypi/v/axioms?style=flat-square)](https://pypi.org/project/axioms/)
[![PyPI downloads](https://img.shields.io/pypi/dm/axioms?style=flat-square)](https://pypi.org/project/axioms/)
[![Python versions](https://img.shields.io/pypi/pyversions/axioms?style=flat-square)](https://pypi.org/project/axioms/)  
[![Issues](https://img.shields.io/github/issues/janheinrichmerker/axioms?style=flat-square)](https://github.com/janheinrichmerker/axioms/issues)
[![Pull requests](https://img.shields.io/github/issues-pr/janheinrichmerker/axioms?style=flat-square)](https://github.com/janheinrichmerker/axioms/pulls)
[![Commit activity](https://img.shields.io/github/commit-activity/m/janheinrichmerker/axioms?style=flat-square)](https://github.com/janheinrichmerker/axioms/commits)
[![License](https://img.shields.io/github/license/janheinrichmerker/axioms?style=flat-square)](LICENSE)

# ↕️ axioms

Axiomatic constraints for information retrieval and retrieval-augmented generation.

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

If you have found an important feature missing from our tool, please suggest it by creating an [issue](https://github.com/janheinrichmerker/axioms/issues). We also gratefully accept [pull requests](https://github.com/janheinrichmerker/axioms/pulls)!

If you are unsure about anything, post an [issue](https://github.com/janheinrichmerker/axioms/issues/new) or contact us:

- [heinrich.merker@uni-jena.de](mailto:heinrich.merker@uni-jena.de)

We are happy to help!

## License

This repository is released under the [MIT license](LICENSE).
Files in the `data/` directory are exempt from this license.
