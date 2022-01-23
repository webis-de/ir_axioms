from typing import Iterable, Union

from ir_axioms import registry
from ir_axioms.axiom import Axiom, AggregatedAxiom

AxiomLike = Union[str, Axiom, Iterable["AxiomLike"]]


def parse_axiom(axiom_name: str) -> Axiom:
    return registry[axiom_name]


def to_axiom(axiom_like: AxiomLike) -> Axiom:
    if isinstance(axiom_like, str):
        return parse_axiom(axiom_like)
    elif isinstance(axiom_like, Iterable):
        return AggregatedAxiom([to_axiom(item) for item in axiom_like])
    else:
        assert isinstance(axiom_like, Axiom)
        return axiom_like
