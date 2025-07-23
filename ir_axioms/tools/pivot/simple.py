from functools import cached_property
from dataclasses import dataclass
from random import Random
from typing import Optional, Sequence, Union, TypeVar

from ir_axioms.tools.pivot.base import PivotSelection

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output")


@dataclass(frozen=True, kw_only=True)
class RandomPivotSelection(PivotSelection[Input, Output]):
    seed: Optional[Union[int, float, str, bytes, bytearray]] = None

    @cached_property
    def _random(self) -> Random:
        return Random(self.seed)  # nosec: B311

    def select_pivot(self, input: Input, vertices: Sequence[Output]) -> Output:
        return vertices[self._random.randint(0, len(vertices) - 1)]


@dataclass(frozen=True, kw_only=True)
class FirstPivotSelection(PivotSelection[Input, Output]):
    def select_pivot(self, input: Input, vertices: Sequence[Output]) -> Output:
        return vertices[0]


@dataclass(frozen=True, kw_only=True)
class LastPivotSelection(PivotSelection[Input, Output]):
    def select_pivot(self, input: Input, vertices: Sequence[Output]) -> Output:
        return vertices[-1]


@dataclass(frozen=True, kw_only=True)
class MiddlePivotSelection(PivotSelection[Input, Output]):
    def select_pivot(self, input: Input, vertices: Sequence[Output]) -> Output:
        return vertices[(len(vertices) - 1) // 2]
