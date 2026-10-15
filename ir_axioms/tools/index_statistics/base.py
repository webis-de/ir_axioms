from collections.abc import Iterable, Mapping
from math import log
from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexStatistics(Protocol):
    @property
    def document_count(self) -> int: ...

    def document_frequency(self, term: str) -> int: ...

    def document_frequencies(self, terms: Iterable[str]) -> Mapping[str, int]:
        return {term: self.document_frequency(term) for term in terms}

    def _inverse_document_frequency(self, document_frequency: int) -> float:
        if document_frequency == 0:
            return 0
        return log(self.document_count / document_frequency)

    def inverse_document_frequency(self, term: str) -> float:
        return self._inverse_document_frequency(self.document_frequency(term))

    def inverse_document_frequencies(self, terms: Iterable[str]) -> Mapping[str, float]:
        return {
            term: self._inverse_document_frequency(document_frequency)
            for term, document_frequency in self.document_frequencies(terms).items()
        }
