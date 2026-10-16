# Contributing

<p class="subtitle">Development setup, checks, and documentation</p>

If you have found an important feature missing from our tool, please suggest it by creating an [issue](https://github.com/webis-de/ir_axioms/issues).
We also gratefully accept [pull requests](https://github.com/webis-de/ir_axioms/pulls)!

## Development Setup

1. Install [Python 3.11](https://python.org/downloads/) or later.
2. Create and activate a virtual environment:

    ```shell
    python3 -m venv venv/
    source venv/bin/activate
    ```

3. Install the project dependencies:

    ```shell
    pip install -e .[tests]
    ```

4. Install the spaCy model:

    ```shell
    python -m spacy download en_core_web_sm
    ```

5. After having implemented a new feature, please check the code format, inspect common LINT errors, and run all unit tests with the following commands:

    ```shell
    ruff check .                   # Code format and LINT
    mypy .                         # Static typing
    pytest .                       # Unit tests
    ```

## Documentation

This documentation is built with [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.
The sources are located in the [`docs/`](https://github.com/webis-de/ir_axioms/tree/main/docs) directory and the configuration in [`mkdocs.yml`](https://github.com/webis-de/ir_axioms/blob/main/mkdocs.yml).

1. Install the documentation dependencies:

    ```shell
    pip install -e .[docs]
    ```

2. Preview the documentation locally (with live reload):

    ```shell
    mkdocs serve
    ```

3. Build the documentation (as in CI, failing on warnings):

    ```shell
    mkdocs build --strict
    ```

The documentation is deployed to [GitHub Pages](https://webis-de.github.io/ir_axioms/) automatically on every push to the `main` branch.

### Citations

All references are stored in [`docs/references.bib`](https://github.com/webis-de/ir_axioms/blob/main/docs/references.bib).
To cite a reference in any page, use `[@key]` or `[@key1; @key2]` (with the BibTeX keys).
The citations are rendered as links to the [references page](../references.md), and unknown keys fail the build.

When adding a reference, please copy its BibTeX entry from [DBLP](https://dblp.org/) if the publication is indexed there.

### Style

The theme follows the colors and rough design of the [Webis lecture notes](https://downloads.webis.de/lecturenotes/lecturenotes-generic/unit-en-lecturenotes-tutorial.pdf) (see [`docs/stylesheets/webis.css`](https://github.com/webis-de/ir_axioms/blob/main/docs/stylesheets/webis.css)).
Start each page with a title and a short subtitle:

```markdown
# Page Title

<p class="subtitle">Short description of the page</p>
```

## License

This repository is released under the [MIT license](https://github.com/webis-de/ir_axioms/blob/main/LICENSE).
Files in the `data/` directory are exempt from this license.
