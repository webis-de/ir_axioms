from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from random import Random
from typing import Optional, Any, Sequence

from ir_axioms.model import Query, RankedDocument, IndexContext


class PivotSelection(ABC):
    @abstractmethod
    def select_pivot(
            self,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        pass


@dataclass(frozen=True)
class RandomPivotSelection(PivotSelection):
    seed: Optional[Any] = None

    @cached_property
    def _random(self) -> Random:
        return Random(self.seed)

    def select_pivot(
            self,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[self._random.randint(0, len(vertices) - 1)]


@dataclass(frozen=True)
class FirstPivotSelection(PivotSelection):
    def select_pivot(
            self,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[0]


@dataclass(frozen=True)
class LastPivotSelection(PivotSelection):
    def select_pivot(
            self,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[-1]


@dataclass(frozen=True)
class MiddlePivotSelection(PivotSelection):
    def select_pivot(
            self,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[(len(vertices) - 1) // 2]
