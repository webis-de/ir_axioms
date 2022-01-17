from random import randint
from typing import List

from ir_axioms import logger
from ir_axioms.axiom import Axiom
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


def _kwiksort(
        axiom: Axiom,
        query: Query,
        context: RerankingContext,
        vertices: List[RankedDocument]
) -> List[RankedDocument]:
    if len(vertices) == 0:
        return []

    vertices_left = []
    vertices_right = []

    # Select random pivot.
    logger.debug("Selecting reranking pivot.")
    pivot = vertices[randint(0, len(vertices) - 1)]

    for vertex in vertices:
        if vertex == pivot:
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

    vertices_left = _kwiksort(
        axiom,
        query,
        context,
        vertices_left
    )
    vertices_right = _kwiksort(
        axiom,
        query,
        context,
        vertices_right
    )

    return [*vertices_left, pivot, *vertices_right]


def _reset_score(ranking: List[RankedDocument]) -> List[RankedDocument]:
    length = len(ranking)
    return [
        RankedDocument(
            id=document.id,
            score=length - i,
            rank=i + 1,
        )
        for i, document in enumerate(ranking)
    ]
