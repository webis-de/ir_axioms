from pytest import mark, warns

from ir_axioms.tools.aspects import YakeAspectExtraction

_TEXT = (
    "Machine learning models need training data. "
    "Deep neural networks are machine learning models. "
    "Neural networks learn from training data."
)


def test_aspects_default() -> None:
    aspects = YakeAspectExtraction().aspects(_TEXT)

    # The top 10 phrases, which include single words and bigrams.
    assert aspects == {
        "datum",
        "learning",
        "machine",
        "model",
        "training",
        "neural",
        "network",
        "deep",
        "training datum",
        "machine learning",
    }


def test_aspects_respect_top_n() -> None:
    assert YakeAspectExtraction(top_n=1).aspects(_TEXT) == {"datum"}
    assert YakeAspectExtraction(top_n=3).aspects(_TEXT) == {
        "datum",
        "learning",
        "machine",
    }


def test_aspects_respect_threshold() -> None:
    # YAKE scores are positive and lower is better.
    assert YakeAspectExtraction(threshold=0.55).aspects(_TEXT) == {
        "datum",
        "learning",
        "machine",
        "model",
    }
    assert YakeAspectExtraction(threshold=0.0).aspects(_TEXT) == set()


def test_aspects_bigrams() -> None:
    extraction = YakeAspectExtraction(ngram_range=(2, 2))

    assert extraction.aspects(_TEXT) == {
        "machine learning",
        "learning model",
        "training datum",
        "deep neural",
        "neural network",
    }


def test_aspects_ngram_range_is_inclusive() -> None:
    extraction = YakeAspectExtraction(ngram_range=(3, 3))

    assert extraction.aspects(_TEXT) == {
        "machine learning model",
        "deep neural network",
    }


def test_aspects_default_ngram_range_includes_trigrams() -> None:
    # Trigrams score worst here, so they only appear beyond the default top 10.
    extraction = YakeAspectExtraction(top_n=20)

    aspects = extraction.aspects(_TEXT)

    assert {"neural", "neural network", "deep neural network"} <= aspects


def test_iter_aspects_matches_aspects() -> None:
    extraction = YakeAspectExtraction()
    texts = [_TEXT, "Paris is the capital of France."]

    aspects = list(extraction.iter_aspects(texts))

    assert aspects == [extraction.aspects(text) for text in texts]
    assert aspects[1] == {"Paris", "France", "capital"}


def test_aspects_empty_text_is_empty() -> None:
    assert YakeAspectExtraction().aspects("") == set()


@mark.parametrize("text", ["the of and", "!!!", "   "])
def test_aspects_without_content_words_is_empty(text: str) -> None:
    with warns(UserWarning, match="Falling back"):
        assert YakeAspectExtraction().aspects(text) == set()


@mark.parametrize("text", ["cat", "cats", "cat cat cat", "what is a cat"])
def test_aspects_single_content_word_falls_back_to_unigram(text: str) -> None:
    with warns(UserWarning, match="Falling back"):
        assert YakeAspectExtraction().aspects(text) == {"cat"}


def test_aspects_single_content_word_ignores_threshold() -> None:
    with warns(UserWarning, match="Falling back"):
        assert YakeAspectExtraction(threshold=0.0).aspects("cat") == {"cat"}


def test_aspects_single_content_word_without_unigrams_is_empty() -> None:
    with warns(UserWarning, match="Falling back"):
        assert YakeAspectExtraction(ngram_range=(2, 3)).aspects("cat") == set()


@mark.filterwarnings("error:Falling back:UserWarning")
def test_aspects_does_not_warn_without_fallback() -> None:
    YakeAspectExtraction().aspects(_TEXT)


def test_iter_aspects_falls_back_per_text() -> None:
    extraction = YakeAspectExtraction()

    with warns(UserWarning, match="Falling back"):
        aspects = list(extraction.iter_aspects(["cat", "the of and", _TEXT]))

    assert aspects == [{"cat"}, set(), extraction.aspects(_TEXT)]
