from math import log

from pytest import approx

from ir_axioms.tools.index_statistics import IndexStatistics


class _FixedIndexStatistics(IndexStatistics):
    @property
    def document_count(self) -> int:
        return 100

    def document_frequency(self, term: str) -> int:
        return {"common": 100, "medium": 10, "rare": 1}.get(term, 0)


def test_inverse_document_frequency() -> None:
    statistics = _FixedIndexStatistics()

    assert statistics.inverse_document_frequency("medium") == approx(log(10))
    assert statistics.inverse_document_frequency("rare") == approx(log(100))


def test_inverse_document_frequency_of_term_in_all_documents_is_zero() -> None:
    assert _FixedIndexStatistics().inverse_document_frequency("common") == 0


def test_inverse_document_frequency_of_unknown_term_is_zero() -> None:
    assert _FixedIndexStatistics().inverse_document_frequency("unknown") == 0


def test_inverse_document_frequency_decreases_with_document_frequency() -> None:
    statistics = _FixedIndexStatistics()

    assert (
        statistics.inverse_document_frequency("rare")
        > statistics.inverse_document_frequency("medium")
        > statistics.inverse_document_frequency("common")
    )
