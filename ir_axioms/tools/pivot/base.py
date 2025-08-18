from typing import Generic, Sequence, Protocol, runtime_checkable, TypeVar

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output")


@runtime_checkable
class PivotSelection(Protocol, Generic[Input, Output]):
    def select_pivot(self, input: Input, vertices: Sequence[Output]) -> Output: ...
