from typing import Sequence

from ir_axioms import logger
from ir_axioms.axiom import Axiom
from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.modules.pivot import RandomPivotSelection, PivotSelection


def kwiksort(
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
    pivot = pivot_selection.select_pivot(query, context, vertices)

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

    vertices_left = kwiksort(
        axiom,
        query,
        context,
        vertices_left
    )
    vertices_right = kwiksort(
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
