from typing import Mapping, Sequence

from pytest import approx

from ir_axioms.model import (
    Document,
    GenerationRequest,
    GenerationResponse,
    Query,
)
from ir_axioms.model.utils import TokenizedString
from ir_axioms.tools.contents import SimpleTextContents
from ir_axioms.tools.text_statistics import (
    DocumentQueryTextStatistics,
    GenerationInputOutputTextStatistics,
    SimpleTextStatistics,
    TextStatistics,
)
from ir_axioms.tools.tokenizer import TermTokenizer


class _WhitespaceTermTokenizer(TermTokenizer):
    def terms(self, text: str) -> Sequence[str]:
        return text.split()


class _FailingTermTokenizer(TermTokenizer):
    def terms(self, text: str) -> Sequence[str]:
        raise AssertionError("Tokenizer should not be called.")


class _FixedTextStatistics(TextStatistics):
    def __init__(self, term_counts: Mapping[str, int]) -> None:
        self._term_counts = term_counts

    def term_counts(self, document) -> Mapping[str, int]:
        return self._term_counts


def _simple_text_statistics() -> SimpleTextStatistics:
    return SimpleTextStatistics(
        text_contents=SimpleTextContents(),
        term_tokenizer=_WhitespaceTermTokenizer(),
    )


def test_term_counts() -> None:
    statistics = _simple_text_statistics()

    term_counts = statistics.term_counts(Document(id="d1", text="a b a c a b"))

    assert dict(term_counts) == {"a": 3, "b": 2, "c": 1}


def test_term_counts_empty_text() -> None:
    statistics = _simple_text_statistics()

    assert dict(statistics.term_counts(Document(id="d1", text=""))) == {}


def test_term_count() -> None:
    statistics = _simple_text_statistics()
    document = Document(id="d1", text="a b a c a b")

    assert statistics.term_count(document, "a") == 3
    assert statistics.term_count(document, "c") == 1
    assert statistics.term_count(document, "missing") == 0


def test_term_frequencies() -> None:
    statistics = _simple_text_statistics()

    term_frequencies = statistics.term_frequencies(Document(id="d1", text="a b a a"))

    assert term_frequencies == approx({"a": 0.75, "b": 0.25})
    assert sum(term_frequencies.values()) == approx(1.0)


def test_term_frequencies_empty_text() -> None:
    statistics = _simple_text_statistics()

    assert dict(statistics.term_frequencies(Document(id="d1", text=""))) == {}


def test_term_frequency() -> None:
    statistics = _simple_text_statistics()
    document = Document(id="d1", text="a b a a")

    assert statistics.term_frequency(document, "a") == approx(0.75)
    assert statistics.term_frequency(document, "missing") == 0


def test_tokenized_string_uses_precomputed_tokens() -> None:
    statistics = SimpleTextStatistics(
        text_contents=SimpleTextContents(),
        term_tokenizer=_FailingTermTokenizer(),
    )
    text = TokenizedString("ignored raw text", tokens={"x": 2, "y": 1})

    assert dict(statistics.term_counts(Document(id="d1", text=text))) == {
        "x": 2,
        "y": 1,
    }


def test_document_query_dispatches_by_type() -> None:
    statistics = DocumentQueryTextStatistics(
        document_text_contents=_FixedTextStatistics({"doc": 1, "term": 3}),
        query_text_contents=_FixedTextStatistics({"query": 2}),
    )
    document = Document(id="d1")
    query = Query(id="q1")

    assert dict(statistics.term_counts(document)) == {"doc": 1, "term": 3}
    assert dict(statistics.term_counts(query)) == {"query": 2}
    assert statistics.term_frequencies(document) == approx({"doc": 0.25, "term": 0.75})
    assert statistics.term_frequencies(query) == approx({"query": 1.0})
    assert statistics.term_count(document, "query") == 0
    assert statistics.term_frequency(query, "query") == approx(1.0)


def _generation_input_output_text_statistics() -> GenerationInputOutputTextStatistics:
    return GenerationInputOutputTextStatistics(
        generation_input_text_statistics=_FixedTextStatistics({"input": 1}),
        generation_output_text_statistics=_FixedTextStatistics({"output": 1}),
    )


def test_generation_request_response_dispatches_by_type() -> None:
    statistics = _generation_input_output_text_statistics()
    request = GenerationRequest(text="")
    response = GenerationResponse(text="")

    assert dict(statistics.term_counts(request)) == {"input": 1}
    assert dict(statistics.term_counts(response)) == {"output": 1}
    assert statistics.term_frequencies(response) == approx({"output": 1.0})
