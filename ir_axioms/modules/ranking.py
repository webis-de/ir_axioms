from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from random import Random
from typing import Optional, Any, Sequence

from ir_axioms import logger
from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Query, RankedDocument, IndexContext


class PivotSelection(ABC):
    @abstractmethod
    def select_pivot(
            self,
            axiom: Axiom,
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
            axiom: Axiom,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[self._random.randint(0, len(vertices) - 1)]


@dataclass(frozen=True)
class FirstPivotSelection(PivotSelection):
    def select_pivot(
            self,
            axiom: Axiom,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[0]


@dataclass(frozen=True)
class LastPivotSelection(PivotSelection):
    def select_pivot(
            self,
            axiom: Axiom,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[-1]


@dataclass(frozen=True)
class MiddlePivotSelection(PivotSelection):
    def select_pivot(
            self,
            axiom: Axiom,
            query: Query,
            context: IndexContext,
            vertices: Sequence[RankedDocument]
    ) -> RankedDocument:
        return vertices[(len(vertices) - 1) // 2]


def kwik_sort(
        axiom: Axiom,
        query: Query,
        context: IndexContext,
        vertices: Sequence[RankedDocument],
        pivot_selection: PivotSelection = RandomPivotSelection(),
) -> Sequence[RankedDocument]:
    if len(vertices) == 0:
        return []

    vertices_left = []
    vertices_right = []

    # Select random pivot.
    logger.debug("Selecting reranking pivot.")
    pivot = pivot_selection.select_pivot(axiom, query, context, vertices)

    for vertex in vertices:
        if vertex is pivot:
            continue

        preference = axiom.preference(context, query, vertex, pivot)
        if preference > 0:
            vertices_left.append(vertex)
        elif preference < 0:
            vertices_right.append(vertex)
        elif vertex.rank < pivot.rank:
            vertices_left.append(vertex)
        elif vertex.rank > pivot.rank:
            vertices_right.append(vertex)
        else:
            raise RuntimeError(
                f"Tie during reranking. "
                f"Document {vertex} has same preference "
                f"and rank as pivot document {pivot}."
            )

    vertices_left = kwik_sort(
        axiom,
        query,
        context,
        vertices_left
    )
    vertices_right = kwik_sort(
        axiom,
        query,
        context,
        vertices_right
    )

    return [*vertices_left, pivot, *vertices_right]


def reset_score(ranking: Sequence[RankedDocument]) -> Sequence[RankedDocument]:
    length = len(ranking)
    return [
        RankedDocument(
            id=document.id,
            score=length - i,
            rank=i + 1,
        )
        for i, document in enumerate(ranking)
    ]
