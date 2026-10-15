from typing import Sequence

from numpy import array
from pytest import approx

from ir_axioms.tools.similarity import (
    AverageTermSimilaritySentenceSimilarity,
    SentenceSimilarity,
    TermSimilarity,
)
from ir_axioms.tools.tokenizer import TermTokenizer

_TERM_SIMILARITIES = {
    frozenset({"cat", "kitten"}): 0.8,
    frozenset({"cat", "dog"}): 0.5,
    frozenset({"dog", "kitten"}): 0.4,
}


class _FixedTermSimilarity(TermSimilarity):
    def similarity(self, term1: str, term2: str) -> float:
        if term1 == term2:
            return 1.0
        return _TERM_SIMILARITIES.get(frozenset({term1, term2}), 0.0)


class _JaccardSentenceSimilarity(SentenceSimilarity):
    def similarity(self, sentence1: str, sentence2: str) -> float:
        terms1 = set(sentence1.split())
        terms2 = set(sentence2.split())
        return len(terms1 & terms2) / len(terms1 | terms2)


class _WhitespaceTermTokenizer(TermTokenizer):
    def terms(self, text: str) -> Sequence[str]:
        return text.split()


def test_term_similarities_matrix() -> None:
    similarities = _FixedTermSimilarity().similarities(["cat", "dog", "kitten"])

    assert similarities.shape == (3, 3)
    assert similarities.diagonal() == approx([1.0, 1.0, 1.0])
    assert similarities == approx(similarities.T)
    assert similarities[0, 2] == approx(0.8)


def test_term_similarity_sums() -> None:
    similarity_sums = _FixedTermSimilarity().similarity_sums(["cat", "dog", "kitten"])

    assert similarity_sums == approx({"cat": 1.3, "dog": 0.9, "kitten": 1.2})


def test_term_similarity_sums_ignores_duplicates() -> None:
    similarity_sums = _FixedTermSimilarity().similarity_sums(["cat", "cat", "dog"])

    # Self-similarity of duplicates must not be counted.
    assert similarity_sums == approx({"cat": 0.5, "dog": 0.5})


def test_term_average_similarity() -> None:
    similarity = _FixedTermSimilarity()

    assert similarity.average_similarity(["cat"], ["kitten", "dog"]) == approx(0.65)


def test_term_average_similarity_empty_is_zero() -> None:
    similarity = _FixedTermSimilarity()

    assert similarity.average_similarity([], ["cat"]) == 0
    assert similarity.average_similarity(["cat"], []) == 0


def test_term_max_similarity_pairs() -> None:
    similarity = _FixedTermSimilarity()

    pairs = similarity.max_similarity_pairs(["cat", "dog"], ["kitten"])

    assert pairs == {("cat", "kitten")}


def test_term_max_similarity_pairs_returns_ties() -> None:
    similarity = _FixedTermSimilarity()

    pairs = similarity.max_similarity_pairs(["cat", "dog"], ["cat", "dog"])

    assert pairs == {("cat", "cat"), ("dog", "dog")}


def test_term_max_similarity_pairs_empty() -> None:
    similarity = _FixedTermSimilarity()

    assert similarity.max_similarity_pairs([], ["cat"]) == set()
    assert similarity.max_similarity_pairs(["cat"], []) == set()


def test_term_max_and_min_average_similarity_terms() -> None:
    similarity = _FixedTermSimilarity()
    terms = ["cat", "dog", "kitten"]

    assert similarity.max_average_similarity_terms(terms) == {"cat"}
    assert similarity.min_average_similarity_terms(terms) == {"dog"}


def test_term_max_and_min_average_similarity_terms_empty() -> None:
    similarity = _FixedTermSimilarity()

    assert similarity.max_average_similarity_terms([]) == set()
    assert similarity.min_average_similarity_terms([]) == set()


def test_sentence_self_similarities() -> None:
    similarities = _JaccardSentenceSimilarity().self_similarities(["a b", "b c", "d"])

    assert similarities.shape == (3, 3)
    assert similarities.diagonal() == approx([1.0, 1.0, 1.0])
    assert similarities == approx(similarities.T)
    assert similarities[0, 1] == approx(1 / 3)
    assert similarities[0, 2] == approx(0.0)


def test_sentence_paired_similarities() -> None:
    similarities = _JaccardSentenceSimilarity().paired_similarities(
        ["a b", "c"],
        ["a b", "b", "d"],
    )

    assert similarities.shape == (2, 3)
    assert similarities == approx(array([[1.0, 0.5, 0.0], [0.0, 0.0, 0.0]]))


def test_sentence_average_similarity() -> None:
    similarity = _JaccardSentenceSimilarity()

    average = similarity.average_similarity(["a b", "c"], ["a b", "b", "d"])

    assert average == approx(1.5 / 6)


def test_average_term_similarity_sentence_similarity() -> None:
    similarity = AverageTermSimilaritySentenceSimilarity(
        term_tokenizer=_WhitespaceTermTokenizer(),
        term_similarity=_FixedTermSimilarity(),
    )

    assert similarity.similarity("cat", "kitten dog") == approx(0.65)
    assert similarity.similarity("cat", "cat") == approx(1.0)
    assert similarity.similarity("", "cat") == 0
