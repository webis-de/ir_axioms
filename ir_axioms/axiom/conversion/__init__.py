from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import isabstract
from typing import Iterable, Union, final, List

from ir_axioms.axiom import Axiom, AggregatedAxiom
from ir_axioms.axiom.cache import _AxiomLRUCache
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext
from ir_axioms import registry

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
