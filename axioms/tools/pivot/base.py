from typing import Generic, Sequence, Protocol, runtime_checkable

from axioms.model import Input, Output


@runtime_checkable
class PivotSelection(Protocol, Generic[Input, Output]):
    def select_pivot(self, input: Input, vertices: Sequence[Output]) -> Output:
        pass
