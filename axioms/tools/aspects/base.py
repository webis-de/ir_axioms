from typing import Protocol, runtime_checkable, Sequence, AbstractSet


@runtime_checkable
class AspectExtraction(Protocol):
    def aspects(self, text: str) -> Sequence[str]:
        pass

    def unique_aspects(self, text: str) -> AbstractSet[str]:
        return set(self.aspects(text))
