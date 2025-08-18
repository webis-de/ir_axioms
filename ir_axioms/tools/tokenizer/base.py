from typing import Sequence, Protocol, runtime_checkable, AbstractSet, Collection


@runtime_checkable
class TermTokenizer(Protocol):
    def terms(self, text: str) -> Sequence[str]: ...

    def terms_unordered(self, text: str) -> Collection[str]:
        return self.terms(text)

    def unique_terms(self, text: str) -> AbstractSet[str]:
        return set(self.terms(text))


@runtime_checkable
class SentenceTokenizer(Protocol):
    def sentences(self, text: str) -> Sequence[str]: ...
