from typing import Sequence

from axioms import logger
from axioms.axiom import Axiom
from axioms.model import Input, Output
from axioms.tools.pivot import RandomPivotSelection, PivotSelection


def kwiksort(
    axiom: Axiom[Input, Output],
    input: Input,
    vertices: Sequence[Output],
    pivot_selection: PivotSelection[Input, Output] = RandomPivotSelection(),
) -> Sequence[Output]:
    if len(vertices) == 0:
        return []

    vertices_left = []
    vertices_right = []

    # Select random pivot.
    logger.debug("Selecting reranking pivot.")
    pivot = pivot_selection.select_pivot(input, vertices)

    for vertex in vertices:
        if vertex is pivot:
            continue

        preference = axiom.preference(input, vertex, pivot)
        if preference > 0:
            vertices_left.append(vertex)
        elif preference < 0:
            vertices_right.append(vertex)
        else:
            raise RuntimeError(
                f"Tie during reranking. "
                f"Document {vertex} has same preference "
                f"and rank as pivot document {pivot}."
            )

    vertices_left_sorted = kwiksort(
        axiom=axiom,
        input=input,
        vertices=vertices_left,
        pivot_selection=pivot_selection,
    )
    vertices_right_sorted = kwiksort(
        axiom=axiom,
        input=input,
        vertices=vertices_right,
        pivot_selection=pivot_selection,
    )

    return [*vertices_left_sorted, pivot, *vertices_right_sorted]
