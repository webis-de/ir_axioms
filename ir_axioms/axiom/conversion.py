from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Union, Type, TypeVar

from ir_axioms import registry
from ir_axioms.axiom.arithmetic import SumAxiom, ProductAxiom, \
    MajorityVoteAxiom, AndAxiom, UniformAxiom
from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext

AxiomLike = Union[str, Axiom, int, float, Iterable["AxiomLike"]]

AggregationAxiom = TypeVar(
    "AggregationAxiom",
    SumAxiom,
    ProductAxiom,
    AndAxiom,
    MajorityVoteAxiom,
)


def to_axiom(
        axiom_like: AxiomLike,
        aggregation: Type[AggregationAxiom] = SumAxiom
) -> Axiom:
    if isinstance(axiom_like, str):
        return registry[axiom_like]
    elif isinstance(axiom_like, (float, int)):
        return UniformAxiom(axiom_like)
    elif isinstance(axiom_like, Iterable):
        return aggregation([to_axiom(item) for item in axiom_like])
    else:
        assert isinstance(axiom_like, Axiom)
        return axiom_like


@dataclass(frozen=True)
class AutoAxiom(Axiom):
    axiom_like: AxiomLike
    aggregation: Type[AggregationAxiom] = SumAxiom

    @cached_property
    def _axiom(self) -> Axiom:
        return to_axiom(self.axiom_like, self.aggregation)

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument,
    ) -> float:
        return self._axiom.preference(context, query, document1, document2)
