from collections.abc import Collection, Iterable, Iterator, Sequence, Set
from typing import Protocol, runtime_checkable


@runtime_checkable
class TermTokenizer(Protocol):
    def terms(self, text: str) -> Sequence[str]: ...

    def terms_unordered(self, text: str) -> Collection[str]:
        return self.terms(text)

    def unique_terms(self, text: str) -> Set[str]:
        return set(self.terms(text))

    def iter_terms(self, texts: Iterable[str]) -> Iterator[Sequence[str]]:
        for text in texts:
            yield self.terms(text)

    def iter_terms_unordered(self, texts: Iterable[str]) -> Iterator[Collection[str]]:
        for text in texts:
            yield self.terms_unordered(text)

    def iter_unique_terms(self, texts: Iterable[str]) -> Iterator[Set[str]]:
        for text in texts:
            yield self.unique_terms(text)


@runtime_checkable
class SentenceTokenizer(Protocol):
    def sentences(self, text: str) -> Sequence[str]: ...

    def iter_sentences(self, texts: Iterable[str]) -> Iterator[Sequence[str]]:
        for text in texts:
            yield self.sentences(text)
