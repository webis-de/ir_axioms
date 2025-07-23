from typing import (
    Protocol,
    runtime_checkable,
    AbstractSet,
    Iterator,
    Iterable,
)


@runtime_checkable
class AspectExtraction(Protocol):
    def aspects(self, text: str) -> AbstractSet[str]:
        pass

    def iter_aspects(self, texts: Iterable[str]) -> Iterator[AbstractSet[str]]:
        for text in texts:
            yield self.aspects(text)
