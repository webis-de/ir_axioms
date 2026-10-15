from collections.abc import Iterable, Iterator
from typing import TypeVar, Generic, Protocol, runtime_checkable


T = TypeVar("T", contravariant=True)


@runtime_checkable
class TextContents(Generic[T], Protocol):
    def contents(self, input: T) -> str: ...

    def iter_contents(self, inputs: Iterable[T]) -> Iterator[str]:
        for input in inputs:
            yield self.contents(input)
