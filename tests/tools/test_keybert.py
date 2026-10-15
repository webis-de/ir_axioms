from pytest import mark

from ir_axioms.tools.aspects import KeyBertAspectExtraction
from ir_axioms.utils.libraries import is_keybert_installed

pytestmark = mark.skipif(
    not is_keybert_installed(),
    reason="KeyBERT is not installed.",
)

_TEXT = (
    "Information retrieval systems rank documents. "
    "Retrieval axioms describe ranking constraints for retrieval systems."
)


def test_aspects_are_ngrams_from_text() -> None:
    aspects = KeyBertAspectExtraction(ngram_range=(1, 2)).aspects(_TEXT)

    assert len(aspects) > 0
    vocabulary = set(_TEXT.lower().replace(".", "").split())
    for aspect in aspects:
        terms = aspect.split()
        assert 1 <= len(terms) <= 2
        assert set(terms) <= vocabulary


def test_aspects_respect_top_n() -> None:
    assert len(KeyBertAspectExtraction(top_n=2).aspects(_TEXT)) <= 2


def test_aspects_respect_threshold() -> None:
    # KeyBERT scores are cosine similarities; higher is better.
    assert KeyBertAspectExtraction(threshold=1.01).aspects(_TEXT) == set()


def test_iter_aspects_matches_aspects() -> None:
    extraction = KeyBertAspectExtraction()
    texts = [_TEXT, "Axiomatic thinking for neural ranking models."]

    assert list(extraction.iter_aspects(texts)) == [
        extraction.aspects(text) for text in texts
    ]
