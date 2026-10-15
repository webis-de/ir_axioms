from ir_axioms.tools.aspects import (
    SpacyEntitiesAspectExtraction,
    SpacyNounChunksAspectExtraction,
)
from ir_axioms.tools.tokenizer import SpacySentenceTokenizer, SpacyTermTokenizer


def test_terms_default_normalization() -> None:
    tokenizer = SpacyTermTokenizer()

    # Stopwords and punctuation are removed, terms are lemmatized and lowercased.
    assert tokenizer.terms("The cats were running quickly.") == [
        "cat",
        "run",
        "quickly",
    ]


def test_terms_without_normalization() -> None:
    tokenizer = SpacyTermTokenizer(
        lemmatize=False,
        lowercase=False,
        remove_punctuation=False,
        remove_stopwords=False,
    )

    assert tokenizer.terms("The Cats were running.") == [
        "The",
        "Cats",
        "were",
        "running",
        ".",
    ]


def test_terms_empty_string_returns_no_terms() -> None:
    assert SpacyTermTokenizer().terms("") == []


def test_sentences_splits_multiple_sentences() -> None:
    tokenizer = SpacySentenceTokenizer()

    sentences = tokenizer.sentences("Hello world. This is a test! Is it working?")

    assert sentences == ["Hello world.", "This is a test!", "Is it working?"]


def test_sentences_empty_string_returns_no_sentences() -> None:
    assert SpacySentenceTokenizer().sentences("") == []


def test_noun_chunks_direct() -> None:
    extraction = SpacyNounChunksAspectExtraction(chunks="direct")

    aspects = extraction.aspects("The quick brown fox jumps over the lazy dog.")

    assert aspects == {"the quick brown fox", "the lazy dog"}


def test_noun_chunks_root() -> None:
    extraction = SpacyNounChunksAspectExtraction(chunks="root")

    aspects = extraction.aspects("The quick brown fox jumps over the lazy dog.")

    assert aspects == {"fox", "dog"}


def test_noun_chunks_both() -> None:
    extraction = SpacyNounChunksAspectExtraction(chunks="both")

    aspects = extraction.aspects("The quick brown fox jumps over the lazy dog.")

    assert aspects == {"the quick brown fox", "the lazy dog", "fox", "dog"}


def test_entities() -> None:
    extraction = SpacyEntitiesAspectExtraction()

    aspects = extraction.aspects("Barack Obama visited Paris in 2015.")

    assert {"Barack Obama", "Paris"} <= aspects


def test_entities_without_entities_is_empty() -> None:
    assert SpacyEntitiesAspectExtraction().aspects("no entities here") == set()


def test_entities_iter_aspects_matches_aspects() -> None:
    extraction = SpacyEntitiesAspectExtraction()
    texts = ["Barack Obama visited Paris.", "No entities here.", "Berlin is big."]

    assert list(extraction.iter_aspects(texts)) == [
        extraction.aspects(text) for text in texts
    ]
