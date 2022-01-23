from typing import Iterable, Union

from ir_axioms import registry
from ir_axioms.axiom.arithmetic import SumAxiom
from ir_axioms.axiom.base import Axiom

AxiomLike = Union[str, Axiom, Iterable["AxiomLike"]]


def to_axiom(axiom_like: AxiomLike) -> Axiom:
    if isinstance(axiom_like, str):
        return registry[axiom_like]
    elif isinstance(axiom_like, Iterable):
        return SumAxiom([to_axiom(item) for item in axiom_like])
    else:
        assert isinstance(axiom_like, Axiom)
        return axiom_like
