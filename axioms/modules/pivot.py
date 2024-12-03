from abc import ABC, abstractmethod
from functools import cached_property
from dataclasses import dataclass
from random import Random
from typing import Generic, Optional, Any, Sequence

from axioms.model import Input, Output


class PivotSelection(ABC, Generic[Input, Output]):
    @abstractmethod
    def select_pivot(
        self,
        input: Input,
        vertices: Sequence[Output]
    ) -> Output:
        pass


@dataclass(frozen=True, kw_only=True)
class RandomPivotSelection(PivotSelection[Any, Any]):
    seed: Optional[Any] = None

    @cached_property
    def _random(self) -> Random:
        return Random(self.seed)  # nosec: B311

    def select_pivot(
        self,
        input: Input,
        vertices: Sequence[Output]
    ) -> Output:
        return vertices[self._random.randint(0, len(vertices) - 1)]


@dataclass(frozen=True, kw_only=True)
class FirstPivotSelection(PivotSelection):
    def select_pivot(
        self,
        input: Input,
        vertices: Sequence[Output]
    ) -> Output:
        return vertices[0]


@dataclass(frozen=True, kw_only=True)
class LastPivotSelection(PivotSelection):
    def select_pivot(
        self,
        input: Input,
        vertices: Sequence[Output]
    ) -> Output:
        return vertices[-1]


@dataclass(frozen=True, kw_only=True)
class MiddlePivotSelection(PivotSelection):
    def select_pivot(
        self,
        input: Input,
        vertices: Sequence[Output]
    ) -> Output:
        return vertices[(len(vertices) - 1) // 2]
