from collections.abc import Iterator, Iterable, Set
from typing import Protocol, runtime_checkable


@runtime_checkable
class AspectExtraction(Protocol):
    def aspects(self, text: str) -> Set[str]: ...

    def iter_aspects(self, texts: Iterable[str]) -> Iterator[Set[str]]:
        for text in texts:
            yield self.aspects(text)
