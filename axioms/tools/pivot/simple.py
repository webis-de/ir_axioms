from functools import cached_property
from dataclasses import dataclass
from random import Random
from typing import Optional, Any, Sequence, Union

from axioms.model import Output
from axioms.tools.pivot.base import PivotSelection


@dataclass(frozen=True, kw_only=True)
class RandomPivotSelection(PivotSelection[Any, Output]):
    seed: Optional[Union[int, float, str, bytes, bytearray]] = None

    @cached_property
    def _random(self) -> Random:
        return Random(self.seed)  # nosec: B311

    def select_pivot(self, input: Any, vertices: Sequence[Output]) -> Output:
        return vertices[self._random.randint(0, len(vertices) - 1)]


@dataclass(frozen=True, kw_only=True)
class FirstPivotSelection(PivotSelection[Any, Output]):
    def select_pivot(self, input: Any, vertices: Sequence[Output]) -> Output:
        return vertices[0]


@dataclass(frozen=True, kw_only=True)
class LastPivotSelection(PivotSelection[Any, Output]):
    def select_pivot(self, input: Any, vertices: Sequence[Output]) -> Output:
        return vertices[-1]


@dataclass(frozen=True, kw_only=True)
class MiddlePivotSelection(PivotSelection[Any, Output]):
    def select_pivot(self, input: Any, vertices: Sequence[Output]) -> Output:
        return vertices[(len(vertices) - 1) // 2]
