from typing import Protocol, Mapping, TypeVar, Generic, runtime_checkable


T = TypeVar("T", contravariant=True)


@runtime_checkable
class TextStatistics(Protocol, Generic[T]):
    def term_counts(self, document: T) -> Mapping[str, int]:
        pass

    def term_count(self, document: T, term: str) -> int:
        return self.term_counts(document).get(term, 0)

    def term_frequencies(self, document: T) -> Mapping[str, float]:
        term_counts = self.term_counts(document)
        document_length = sum(term_counts.values())
        return {term: count / document_length for term, count in term_counts.items()}

    def term_frequency(self, document: T, term: str) -> float:
        return self.term_frequencies(document).get(term, 0)
