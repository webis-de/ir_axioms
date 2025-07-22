from math import log
from typing import Protocol, runtime_checkable


@runtime_checkable
class IndexStatistics(Protocol):
    @property
    def document_count(self) -> int: ...

    def document_frequency(self, term: str) -> int: ...

    def inverse_document_frequency(self, term: str) -> float:
        document_frequency = self.document_frequency(term)
        if document_frequency == 0:
            return 0
        return log(self.document_count / document_frequency)
