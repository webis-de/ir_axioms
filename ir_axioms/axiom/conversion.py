from typing import Iterable, Union

from ir_axioms import registry
from ir_axioms.axiom.arithmetic import SumAxiom
from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext

AxiomLike = Union[str, Axiom, Iterable["AxiomLike"]]


def to_axiom(axiom_like: AxiomLike) -> Axiom:
    if isinstance(axiom_like, str):
        return registry[axiom_like]
    elif isinstance(axiom_like, Iterable):
        return SumAxiom([to_axiom(item) for item in axiom_like])
    else:
        assert isinstance(axiom_like, Axiom)
        return axiom_like


class AutoAxiom(Axiom):
    _axiom: Axiom

    def __init__(self, axiom_like: AxiomLike):
        self._axiom = to_axiom(axiom_like)

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument,
    ) -> float:
        return self._axiom.preference(context, query, document1, document2)
