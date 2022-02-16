from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

from ir_axioms import registry
from ir_axioms.axiom.arithmetic import UniformAxiom
from ir_axioms.axiom.base import Axiom, AxiomLike
from ir_axioms.model import Query, RankedDocument, IndexContext


def to_axiom(axiom_like: AxiomLike) -> Axiom:
    if isinstance(axiom_like, Axiom):
        return axiom_like
    elif isinstance(axiom_like, (float, int)):
        return UniformAxiom(axiom_like)
    elif isinstance(axiom_like, str):
        return registry[axiom_like]
    else:
        raise ValueError(f"Cannot convert to axiom: {axiom_like}")


def to_axioms(axiom_likes: Sequence[AxiomLike]) -> Sequence[Axiom]:
    return [to_axiom(axiom_like) for axiom_like in axiom_likes]


@dataclass(frozen=True)
class AutoAxiom(Axiom):
    axiom_like: AxiomLike

    @cached_property
    def _axiom(self) -> Axiom:
        return to_axiom(self.axiom_like)

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument,
    ) -> float:
        return self._axiom.preference(context, query, document1, document2)
