from ir_axioms.tools.tokenizer.blingfire import (
    BlingfireSentenceTokenizer,
    BlingfireTermTokenizer,
)


def test_terms_splits_words() -> None:
    tokenizer = BlingfireTermTokenizer()

    terms = tokenizer.terms("Hello, World! This is a test.")

    assert terms == ["Hello", "World", "This", "is", "a", "test"]


def test_terms_splits_words_with_punctuation() -> None:
    tokenizer = BlingfireTermTokenizer(remove_punctuation=False)

    terms = tokenizer.terms("Hello, World! This is a test.")

    assert terms == ["Hello", ",", "World", "!", "This", "is", "a", "test", "."]


def test_terms_empty_string_returns_no_terms() -> None:
    tokenizer = BlingfireTermTokenizer()

    assert tokenizer.terms("") == []


def test_terms_unordered_matches_terms() -> None:
    tokenizer = BlingfireTermTokenizer()

    text = "the cat sat on the mat"

    assert sorted(tokenizer.terms_unordered(text)) == sorted(tokenizer.terms(text))


def test_unique_terms_deduplicates() -> None:
    tokenizer = BlingfireTermTokenizer()

    unique_terms = tokenizer.unique_terms("the the the cat sat.")

    assert unique_terms == {"the", "cat", "sat"}


def test_sentences_splits_multiple_sentences() -> None:
    tokenizer = BlingfireSentenceTokenizer()

    sentences = tokenizer.sentences("Hello world. This is a test! Is it working?")

    assert sentences == [
        "Hello world.",
        "This is a test!",
        "Is it working?",
    ]


def test_sentences_single_sentence_returns_one_element() -> None:
    tokenizer = BlingfireSentenceTokenizer()

    assert tokenizer.sentences("Just one sentence.") == ["Just one sentence."]


def test_sentences_empty_string_returns_no_sentences() -> None:
    tokenizer = BlingfireSentenceTokenizer()

    assert tokenizer.sentences("") == []
