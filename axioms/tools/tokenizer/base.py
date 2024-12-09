from typing import Sequence, Protocol, runtime_checkable, AbstractSet


@runtime_checkable
class TermTokenizer(Protocol):
    def terms(self, text: str) -> Sequence[str]:
        pass

    def unique_terms(self, text: str) -> AbstractSet[str]:
        return set(self.terms(text))
