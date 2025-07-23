from typing import TypeVar, Generic, Protocol, runtime_checkable


T = TypeVar("T", contravariant=True)


@runtime_checkable
class TextContents(Generic[T], Protocol):
    def contents(self, input: T) -> str:
        pass
