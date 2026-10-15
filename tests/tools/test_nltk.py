from ir_axioms.tools.tokenizer import NltkSentenceTokenizer, NltkTermTokenizer


def test_terms_splits_words_and_punctuation() -> None:
    tokenizer = NltkTermTokenizer()

    terms = tokenizer.terms("Hello, World! This is a test.")

    assert terms == ["Hello", ",", "World", "!", "This", "is", "a", "test", "."]


def test_terms_lowercase() -> None:
    tokenizer = NltkTermTokenizer(lowercase=True)

    assert tokenizer.terms("Hello World") == ["hello", "world"]


def test_terms_lemmatize() -> None:
    tokenizer = NltkTermTokenizer(lemmatize=True)

    assert tokenizer.terms("cats and geese") == ["cat", "and", "goose"]


def test_terms_lemmatize_and_lowercase() -> None:
    tokenizer = NltkTermTokenizer(lemmatize=True, lowercase=True)

    assert tokenizer.terms("Cats and Geese") == ["cat", "and", "goose"]


def test_terms_empty_string_returns_no_terms() -> None:
    assert NltkTermTokenizer().terms("") == []


def test_unique_terms_deduplicates() -> None:
    tokenizer = NltkTermTokenizer(lowercase=True)

    assert tokenizer.unique_terms("The cat and the hat") == {"the", "cat", "and", "hat"}


def test_sentences_splits_multiple_sentences() -> None:
    tokenizer = NltkSentenceTokenizer()

    sentences = tokenizer.sentences("Hello world. This is a test! Is it working?")

    assert sentences == ["Hello world.", "This is a test!", "Is it working?"]


def test_sentences_does_not_split_abbreviations() -> None:
    tokenizer = NltkSentenceTokenizer()

    sentences = tokenizer.sentences("Dr. Smith arrived. He was late.")

    assert sentences == ["Dr. Smith arrived.", "He was late."]


def test_sentences_empty_string_returns_no_sentences() -> None:
    assert NltkSentenceTokenizer().sentences("") == []
