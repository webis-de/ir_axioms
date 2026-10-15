from ir_axioms.tools.similarity import WordNetSynonymSetTermSimilarity


def test_similarity_related_higher_than_unrelated() -> None:
    similarity = WordNetSynonymSetTermSimilarity()

    assert similarity.similarity("dog", "cat") > similarity.similarity("dog", "car")


def test_similarity_identical_term_is_maximal() -> None:
    similarity = WordNetSynonymSetTermSimilarity()

    assert similarity.similarity("dog", "dog") >= similarity.similarity("dog", "cat")


def test_similarity_is_symmetric() -> None:
    similarity = WordNetSynonymSetTermSimilarity()

    assert similarity.similarity("dog", "cat") == similarity.similarity("cat", "dog")


def test_similarity_unknown_term_is_zero() -> None:
    similarity = WordNetSynonymSetTermSimilarity()

    assert similarity.similarity("dog", "xqzvbn") == 0


def test_similarity_is_bounded() -> None:
    similarity = WordNetSynonymSetTermSimilarity(smoothing=2)

    for term1, term2 in [("dog", "cat"), ("car", "automobile"), ("run", "idea")]:
        assert 0 <= similarity.similarity(term1, term2) <= 1
