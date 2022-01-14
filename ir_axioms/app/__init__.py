from random import randint
from typing import List

from ir_axioms import logger
from ir_axioms.axiom import Axiom, AxiomLike
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
            content=document.content,
            score=length - i,
            rank=i + 1,
        )
        for i, document in enumerate(ranking)
    ]


def rerank(
        axiom: AxiomLike,
        context: RerankingContext,
        query: Query,
        ranking: List[RankedDocument],
) -> List[RankedDocument]:
    ranking = _kwiksort(axiom, query, context, ranking)
    ranking = _reset_score(ranking)
    return ranking


def _is_permutated(
        axiom: AxiomLike,
        context: RerankingContext,
        query: Query,
        document_1: RankedDocument,
        document_2: RankedDocument
):
    if document_1 is document_2:
        return False
    preference = axiom.preference(context, query, document_1, document_2)
    if preference == 0 and document_1.rank == document_2.rank:
        return False
    elif preference > 0 and document_1.rank < document_2.rank:
        return False
    elif preference < 0 and document_1.rank > document_2.rank:
        return False
    else:
        return True


def permutations(
        axiom: AxiomLike,
        context: RerankingContext,
        query: Query,
        ranking: List[RankedDocument],
) -> List[List[bool]]:
    return [
        [
            _is_permutated(axiom, context, query, document1, document2)
            if index1 != index2 else False
            for index2, document2 in enumerate(ranking)
        ]
        for index1, document1 in enumerate(ranking)
    ]


def permutation_count(
        axiom: AxiomLike,
        context: RerankingContext,
        query: Query,
        ranking: List[RankedDocument],
) -> List[int]:
    return [
        sum(1 for is_pair_permutated in pairs if is_pair_permutated)
        for pairs in permutations(axiom, context, query, ranking)
    ]


def permutation_frequency(
        axiom: AxiomLike,
        context: RerankingContext,
        query: Query,
        ranking: List[RankedDocument],
) -> List[float]:
    return [
        (count - 1) / len(ranking) if len(ranking) > 0 else 0
        for count in permutation_count(axiom, context, query, ranking)
    ]
