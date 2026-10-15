from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pytest import fixture, mark

from ir_axioms.utils.libraries import is_pyterrier_installed


# Skip if optional dependency is not installed.
pytestmark = mark.skipif(
    not is_pyterrier_installed(),
    reason="PyTerrier is not installed.",
)


_TEXT = "The Running dogs, aren't jumping!"
_DOCUMENTS = [
    {"docno": "d1", "text": _TEXT},
    {"docno": "d2", "text": "Cats are sleeping in the sun."},
]

_TOKENIZATION_PROPERTY_NAMES = ("tokeniser", "termpipelines")


@fixture(autouse=True)
def application_setup() -> Iterator[Any]:
    """
    Start the JVM and restore the tokenization-related ``ApplicationSetup`` properties after each test (they are global JVM state).
    """

    from pyterrier.java import autoclass, init, started

    if not started():
        init()

    application_setup = autoclass("org.terrier.utility.ApplicationSetup")
    properties = application_setup.getProperties()

    # Save the original application setup properties.
    original = {
        name: properties.getProperty(name) for name in _TOKENIZATION_PROPERTY_NAMES
    }

    # Clear tokenization-related properties.
    for name in _TOKENIZATION_PROPERTY_NAMES:
        properties.remove(name)

    yield application_setup

    # Restore the original application setup properties.
    for name, value in original.items():
        if value is None:
            properties.remove(name)
        else:
            properties.setProperty(name, value)


@fixture
def default_index(tmp_path: Path) -> Path:
    from pyterrier.terrier import TerrierIndex

    index_path = tmp_path / "default-index"
    index_path.mkdir()

    # English tokeniser with Terrier's default term pipeline (Stopwords,PorterStemmer).
    TerrierIndex(index_path).indexer(
        meta={"docno": 20, "text": 100},
    ).index(_DOCUMENTS)

    return index_path


@fixture
def plain_index(tmp_path: Path) -> Path:
    from pyterrier.terrier import TerrierIndex

    index_path = tmp_path / "plain-index"
    index_path.mkdir()

    # English tokeniser without any term pipeline.
    TerrierIndex(index_path).indexer(
        meta={"docno": 20, "text": 100},
        stemmer=None,
        stopwords=None,
    ).index(_DOCUMENTS)

    return index_path


def test_default_settings() -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # English tokeniser, no term pipeline.
    tokenizer = TerrierTermTokenizer()

    assert tokenizer.terms(_TEXT) == [
        "the",
        "running",
        "dogs",
        "aren",
        "t",
        "jumping",
    ]


def test_default_settings_unique_terms() -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # English tokeniser, no term pipeline.
    tokenizer = TerrierTermTokenizer()

    assert tokenizer.unique_terms("the dog, the cat") == {
        "the",
        "dog",
        "cat",
    }


def test_explicit_tokeniser() -> None:
    from pyterrier.terrier import TerrierTokeniser

    from ir_axioms.tools import TerrierTermTokenizer

    # Whitespace tokeniser, no term pipeline.
    tokenizer = TerrierTermTokenizer(tokeniser=TerrierTokeniser.whitespace)

    assert tokenizer.terms(_TEXT) == [
        "The",
        "Running",
        "dogs,",
        "aren't",
        "jumping!",
    ]


def test_explicit_java_tokeniser() -> None:
    from pyterrier.java import autoclass

    from ir_axioms.tools import TerrierTermTokenizer

    # Direct reference to WhitespaceTokeniser class, no term pipeline.
    java_tokeniser = autoclass(
        "org.terrier.indexing.tokenisation.WhitespaceTokeniser"
    )()
    tokenizer = TerrierTermTokenizer(tokeniser=java_tokeniser)

    assert tokenizer.terms(_TEXT) == [
        "The",
        "Running",
        "dogs,",
        "aren't",
        "jumping!",
    ]


def test_application_setup_term_pipelines(application_setup: Any) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # Override term pipelines globally to use stop words and Porter stemmer.
    application_setup.setProperty("termpipelines", "Stopwords,PorterStemmer")
    tokenizer = TerrierTermTokenizer()

    assert tokenizer.terms(_TEXT) == [
        "run",
        "dog",
        "aren",
        "jump",
    ]


def test_application_setup_single_term_pipeline(application_setup: Any) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # Override term pipelines globally to use just Porter stemmer.
    application_setup.setProperty("termpipelines", "PorterStemmer")
    tokenizer = TerrierTermTokenizer()

    assert tokenizer.terms(_TEXT) == [
        "the",
        "run",
        "dog",
        "aren",
        "t",
        "jump",
    ]


def test_application_setup_empty_term_pipelines(application_setup: Any) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # Override term pipelines globally to use no post-processing.
    application_setup.setProperty("termpipelines", "")
    tokenizer = TerrierTermTokenizer()

    assert tokenizer.terms(_TEXT) == [
        "the",
        "running",
        "dogs",
        "aren",
        "t",
        "jumping",
    ]


def test_application_setup_tokeniser(application_setup: Any) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # Override tokeniser globally to use WhitespaceTokeniser class (simple class name).
    application_setup.setProperty("tokeniser", "WhitespaceTokeniser")
    tokenizer = TerrierTermTokenizer()

    assert tokenizer.terms(_TEXT) == [
        "The",
        "Running",
        "dogs,",
        "aren't",
        "jumping!",
    ]


def test_application_setup_tokeniser_fully_qualified(application_setup: Any) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # Override tokeniser globally to use WhitespaceTokeniser class (fully qualified class name).
    application_setup.setProperty(
        "tokeniser", "org.terrier.indexing.tokenisation.WhitespaceTokeniser"
    )
    tokenizer = TerrierTermTokenizer()

    assert tokenizer.terms(_TEXT) == [
        "The",
        "Running",
        "dogs,",
        "aren't",
        "jumping!",
    ]


def test_explicit_tokeniser_overrides_application_setup(
    application_setup: Any,
) -> None:
    from pyterrier.terrier import TerrierTokeniser

    from ir_axioms.tools import TerrierTermTokenizer

    # Explicit tokeniser is used, but the term pipeline is still read from the setup.
    application_setup.setProperty("tokeniser", "WhitespaceTokeniser")
    application_setup.setProperty("termpipelines", "PorterStemmer")
    tokenizer = TerrierTermTokenizer(tokeniser=TerrierTokeniser.english)

    assert tokenizer.terms(_TEXT) == [
        "the",
        "run",
        "dog",
        "aren",
        "t",
        "jump",
    ]


def test_index_term_pipelines(default_index: Path) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # English tokeniser with Terrier's default term pipeline (Stopwords,PorterStemmer).
    tokenizer = TerrierTermTokenizer(index_location=default_index)

    assert tokenizer.terms(_TEXT) == [
        "run",
        "dog",
        "aren",
        "jump",
    ]


def test_index_without_term_pipelines(plain_index: Path) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # English tokeniser without any term pipeline.
    tokenizer = TerrierTermTokenizer(index_location=plain_index)

    assert tokenizer.terms(_TEXT) == [
        "the",
        "running",
        "dogs",
        "aren",
        "t",
        "jumping",
    ]


def test_index_term_pipelines_override_application_setup(
    application_setup: Any,
    default_index: Path,
    plain_index: Path,
) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    # English tokeniser with Terrier's default term pipeline (Stopwords,PorterStemmer), overriding empty term pipelines from application setup.
    application_setup.setProperty("termpipelines", "")
    tokenizer = TerrierTermTokenizer(index_location=default_index)
    assert tokenizer.terms(_TEXT) == [
        "run",
        "dog",
        "aren",
        "jump",
    ]

    # English tokeniser without any term pipeline, overriding application setup term pipeline.
    application_setup.setProperty("termpipelines", "Stopwords,PorterStemmer")
    tokenizer = TerrierTermTokenizer(index_location=plain_index)
    assert tokenizer.terms(_TEXT) == [
        "the",
        "running",
        "dogs",
        "aren",
        "t",
        "jumping",
    ]


# Note: Terrier does not persist the tokeniser by default, so we can't test for it here.


def test_index_location_types(default_index: Path) -> None:
    from pyterrier.terrier import TerrierIndex

    from ir_axioms.tools import TerrierTermTokenizer

    for index_location in (
        default_index,
        str(default_index),
        TerrierIndex.coerce(default_index),  # type: ignore
    ):
        tokenizer = TerrierTermTokenizer(index_location=index_location)
        assert tokenizer.terms(_TEXT) == ["run", "dog", "aren", "jump"]


def _lexicon_terms(index_path: Path) -> set[str]:
    from pyterrier.terrier import TerrierIndex

    lexicon = TerrierIndex.coerce(index_path).lexicon()  # type: ignore
    return {str(entry.getKey()) for entry in lexicon}


def test_index_terms_match_lexicon(
    default_index: Path,
    plain_index: Path,
) -> None:
    from ir_axioms.tools import TerrierTermTokenizer

    for index_path in (default_index, plain_index):
        tokenizer = TerrierTermTokenizer(index_location=index_path)
        terms = {
            term
            for document in _DOCUMENTS
            for term in tokenizer.terms(document["text"])
        }
        assert terms == _lexicon_terms(index_path)
