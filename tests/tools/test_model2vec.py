from pytest import approx, mark

from ir_axioms.tools.similarity.model2vec import Model2VecSentenceSimilarity
from ir_axioms.utils.libraries import is_model2vec_installed

pytestmark = mark.skipif(
    not is_model2vec_installed(),
    reason="Model2Vec is not installed.",
)


def test_similarity_identical_sentences_is_one() -> None:
    similarity = Model2VecSentenceSimilarity()

    assert similarity.similarity(
        "The cat sat on the mat.", "The cat sat on the mat."
    ) == approx(1.0)


def test_similarity_related_higher_than_unrelated() -> None:
    similarity = Model2VecSentenceSimilarity()

    related = similarity.similarity("A cat sits on a mat.", "A kitten rests on a rug.")
    unrelated = similarity.similarity(
        "A cat sits on a mat.", "Stock markets fell sharply today."
    )

    assert related > unrelated


def test_self_similarities_shape_and_diagonal() -> None:
    similarity = Model2VecSentenceSimilarity()

    sentences = ["Hello world.", "This is a test.", "Another sentence."]
    similarities = similarity.self_similarities(sentences)

    assert similarities.shape == (3, 3)
    assert similarities.diagonal() == approx([1.0, 1.0, 1.0])
    assert similarities == approx(similarities.T)


def test_paired_similarities_shape() -> None:
    similarity = Model2VecSentenceSimilarity()

    similarities = similarity.paired_similarities(
        ["Hello world.", "This is a test."],
        ["Hello there.", "A test.", "Something else."],
    )

    assert similarities.shape == (2, 3)
