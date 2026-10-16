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

This documentation is built with [MkDocs](https://www.mkdocs.org/), the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme, and the [`mkdocs-webis`](https://github.com/webis-de/mkdocs-webis) plugin for the Webis style and citations.
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

The documentation is deployed to [GitHub Pages](https://ir-axioms.webis.de/) automatically on every release.

### Citations

All references are stored in [`docs/references.bib`](https://github.com/webis-de/ir_axioms/blob/main/docs/references.bib) and rendered by the [`mkdocs-webis`](https://github.com/webis-de/mkdocs-webis) plugin.
To cite a reference in any page, use `[@key]` or `[@key1; @key2]` (with the BibTeX keys).
The citations are rendered as author–year links, and each page that cites a reference automatically gets a "References" section at the bottom (including the BibTeX entries).
Unknown keys fail the build.

When adding a reference, please copy its BibTeX entry from [DBLP](https://dblp.org/) if the publication is indexed there.

### Style

The [`mkdocs-webis`](https://github.com/webis-de/mkdocs-webis) plugin provides the colors and rough design of the [Webis lecture notes](https://downloads.webis.de/lecturenotes/lecturenotes-generic/unit-en-lecturenotes-tutorial.pdf).
Start each page with a title and a short subtitle, and use badges for axiom names:

```markdown
# Page Title

<p class="subtitle">Short description of the page</p>

<span class="badge">TFC1</span> `Tfc1Axiom` — _"Intuition of the axiom."_ [@key]
```

## License

This repository is released under the [MIT license](https://github.com/webis-de/ir_axioms/blob/main/LICENSE).
Files in the `data/` directory are exempt from this license.
