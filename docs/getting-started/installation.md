# Installation

<p class="subtitle">Install <code>ir_axioms</code> from PyPI</p>

The `ir_axioms` framework requires [Python 3.11](https://python.org/downloads/) or later.

For a basic, minimal setup, just install the [`ir_axioms` package from PyPI](https://pypi.org/project/ir-axioms/), and a spaCy model for tokenization:

```shell
pip install ir_axioms
python -m spacy download en_core_web_sm
```

If you do not need any "fancy" extras, just continue to the [quickstart](quickstart.md).

## Optional Dependencies

Some features of `ir_axioms` require additional dependencies that you can install as "extras" of the PyPI package like so:

```shell
pip install ir_axioms[pyterrier,sbert]
```

All available extras are listed here for quick reference:

| Extra | Description |
| :-- | :-- |
| `pyserini` | Use index statistics, document contents, and tokenization from [Pyserini](http://pyserini.io/)/[Anserini](http://anserini.io/) indices. |
| `pyterrier` | Use index statistics, document contents, and tokenization from [PyTerrier](https://pyterrier.readthedocs.io/en/latest/)/[Terrier](https://terrier.org/) indices. Apply KwikSort re-ranking to PyTerrier pipelines. Run axiomatic analyses. |
| `sbert` | Use [Sentence Transformers](https://www.sbert.net/) for sentence similarity (enabled automatically if installed). |
| `model2vec` | Use [Model2Vec](https://minish.ai/packages/model2vec/introduction/) static embeddings for sentence similarity (enabled automatically if installed). |
| `blingfire` | Use [BlingFire](https://github.com/microsoft/BlingFire) for fast tokenization (enabled automatically if installed). |
| `fasttext` | Use [fastText](https://fasttext.cc/) word vectors for term similarity (enabled automatically if installed). |
| `keybert` | Use [KeyBERT](https://maartengr.github.io/KeyBERT/) for aspect extraction (must be enabled manually if installed). |
| `autojudge` | Use axioms as [TREC AutoJudge](https://trec-auto-judge.cs.unh.edu/) evaluators. |
| `experiments` | Dependencies for the [experiment notebooks](../walkthroughs/). |

!!! tip "Reproducing our ICTIR 2025 results"

    To reproduce the results from our ICTIR 2025 paper on axioms for RAG [@merker2025axioms], please install these extras:

    ```shell
    pip install ir_axioms[pyterrier,keybert,sbert,fasttext]
    ```

## Command Line Interface

Installing the `ir_axioms` package also installs the `ir_axioms` CLI command (aliases: `ir-axioms`, `axioms`). Run its help command to list available sub-commands:

```shell
ir_axioms --help
```

!!! note "Work in Progress"

    Right now, the CLI does not do a lot... see the [roadmap](../development/roadmap.md#integrations-and-tools). 

## Development Installation

To work on `ir_axioms` core code itself, please follow the [contribution guide](../development/contributing.md).
