from typing import Generic, TypeVar, Protocol, runtime_checkable


from axioms.model.generation import Aspects

T = TypeVar("T", contravariant=True)


@runtime_checkable
class AspectExtraction(Generic[T], Protocol):
    def extract_aspects(self, input: T) -> Aspects:
        pass
