from typing import Protocol, Mapping, TypeVar, Generic, runtime_checkable


T = TypeVar("T", contravariant=True)


@runtime_checkable
class TextStatistics(Protocol, Generic[T]):
    def term_frequencies(self, document: T) -> Mapping[str, int]:
        pass

    def term_frequency(self, document: T, term: str) -> int:
        return self.term_frequencies(document).get(term, 0)
