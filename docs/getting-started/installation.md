# Installation

<p class="subtitle">Install <code>ir_axioms</code> from PyPI</p>

`ir_axioms` requires [Python 3.11](https://python.org/downloads/) or later.

1. Install the package from PyPI:

    ```shell
    pip install "ir_axioms>=1.0"
    ```

2. Install the spaCy model used for sentence splitting, noun chunks, and entities:

    ```shell
    python -m spacy download en_core_web_sm
    ```

## Optional Dependencies

Some features require additional dependencies that you can install as "extras", e.g., `pip install "ir_axioms[pyterrier,sbert]"`.

| Extra | Description |
|:--|:--|
| `pyserini` | Use index statistics, document contents, and tokenization from Pyserini/Anserini indices. |
| `pyterrier` | Use index statistics, document contents, and tokenization from PyTerrier/Terrier indices. Apply KwikSort re-ranking to PyTerrier pipelines. Run axiomatic analyses. |
| `keybert` | Use KeyBERT for aspect extraction (must be enabled manually if installed). |
| `sbert` | Use Sentence Transformers for sentence similarity (enabled automatically if installed). |
| `model2vec` | Use Model2Vec static embeddings for sentence similarity (enabled automatically if installed). |
| `blingfire` | Use BlingFire for fast tokenization (enabled automatically if installed). |
| `fasttext` | Use fastText word vectors for term similarity (enabled automatically if installed). |
| `autojudge` | Use the TREC AutoJudge integration. |
| `experiments` | Dependencies of the experiment notebooks (e.g., Jupyter, LightGBM, seaborn). |

!!! tip "Reproducing our ICTIR 2025 results"

    To reproduce the results from our ICTIR 2025 paper [@merker2025axioms], please install these extras: `pyterrier,keybert,sbert,fasttext`.

## Command Line Interface

Installing the package also installs the `ir_axioms` command (aliases: `ir-axioms`, `axioms`):

```shell
ir_axioms --help
```

## Development Installation

To work on `ir_axioms` itself, please follow the [contribution guide](../development/contributing.md).
