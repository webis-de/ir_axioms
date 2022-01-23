from abc import ABC, abstractmethod
from inspect import isabstract
from typing import final, List, Union

from ir_axioms import registry
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class Axiom(ABC):
    """
    Base class for all axioms.
    Implements the various operators ``+``, ``-``, ``*``, ``/`` ``|`` ``&``
    as well as ``search()`` for executing a single query and ``compile()`` for rewriting complex pipelines into more simples ones.
    """

    name: str = None
    """
    The axiom's 
    """

    def __init_subclass__(cls, **kwargs):
        # Automatically register this subclass to the global axiom registry.
        if not isabstract(cls) and cls.name is not None:
            registry[cls.name] = cls

    @abstractmethod
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        pass

    @final
    def __add__(self, other: Union["Axiom", float]) -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import SumAxiom
            return SumAxiom([self, other])
        elif isinstance(other, float):
            from ir_axioms.axiom.arithmetic import UniformAxiom
            return self + UniformAxiom(other)
        else:
            return NotImplemented

    @final
    def __radd__(self, other: Union["Axiom", float]) -> "Axiom":
        return self + other

    @final
    def __sub__(self, other: Union["Axiom", float]) -> "Axiom":
        return self + -other

    @final
    def __rsub__(self, other: Union["Axiom", float]) -> "Axiom":
        return -self + other

    @final
    def __mul__(self, other: Union["Axiom", float]) -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import ProductAxiom
            return ProductAxiom([self, other])
        elif isinstance(other, float):
            from ir_axioms.axiom.arithmetic import UniformAxiom
            return self * UniformAxiom(other)
        else:
            return NotImplemented

    @final
    def __rmul__(self, other: Union["Axiom", float]) -> "Axiom":
        return self * other

    @final
    def __truediv__(self, other: Union["Axiom", float]) -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import InvertedAxiom
            return self * InvertedAxiom(other)
        elif isinstance(other, float):
            return self * (1 / other)
        else:
            return NotImplemented

    @final
    def __rtruediv__(self, other: Union["Axiom", float]) -> "Axiom":
        from ir_axioms.axiom.arithmetic import InvertedAxiom
        return InvertedAxiom(self) * other

    @final
    def __mod__(self, other: Union["Axiom", float]) -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import MajorityVoteAxiom
            return MajorityVoteAxiom([self, other])
        elif isinstance(other, float):
            from ir_axioms.axiom.arithmetic import UniformAxiom
            return self % UniformAxiom(other)
        else:
            return NotImplemented

    @final
    def __rmod__(self, other: Union["Axiom", float]) -> "Axiom":
        return self % other

    @final
    def __and__(self, other: Union["Axiom", float]) -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import AndAxiom
            return AndAxiom([self, other])
        elif isinstance(other, float):
            from ir_axioms.axiom.arithmetic import UniformAxiom
            return self & UniformAxiom(other)
        else:
            return NotImplemented

    @final
    def __rand__(self, other: "Axiom") -> "Axiom":
        return self & other

    @final
    def __neg__(self) -> "Axiom":
        from ir_axioms.axiom.arithmetic import NegatedAxiom
        return NegatedAxiom(self)

    @final
    def __pos__(self) -> "Axiom":
        """
        Return the normalized preference of this axiom,
        replacing positive values with 1 and negative values with -1.
        """
        from ir_axioms.axiom.arithmetic import NormalizedAxiom
        return NormalizedAxiom(self)

    @final
    def __invert__(self) -> "Axiom":
        """
        Cache this axiom's preferences, meaning the ``preference()`` method
        will only be called if the context-query-documents tuple
        does not already exist in the cache.
        """
        from ir_axioms.axiom.cache import CachedAxiom
        return CachedAxiom(self)

    @final
    def weighted(self, weight: float) -> "Axiom":
        return self * weight

    @final
    def normalized(self) -> "Axiom":
        """
        Return the normalized preference of this axiom,
        replacing positive values with 1 and negative values with -1.
        """
        return +self

    @final
    def cached(self, capacity: int = 4096) -> "Axiom":
        """
        Cache this axiom's preferences, meaning the ``preference()`` method
        will only be called if the context-query-documents tuple
        does not already exist in the cache.
        """
        from ir_axioms.axiom.cache import CachedAxiom
        return CachedAxiom(self, capacity)

    @final
    def rerank(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[RankedDocument]:
        from ir_axioms.axiom.utils_actions import _kwiksort, _reset_score

        ranking = _kwiksort(self, query, context, ranking)
        ranking = _reset_score(ranking)
        return ranking

    @final
    def preferences(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[List[float]]:
        return [
            [
                self.preference(context, query, document1, document2)
                for document2 in ranking
            ]
            for document1 in ranking
        ]

    @final
    def is_permutated(
            self,
            context: RerankingContext,
            query: Query,
            document_1: RankedDocument,
            document_2: RankedDocument
    ):
        if document_1 is document_2:
            return False
        preference = self.preference(context, query, document_1, document_2)
        if preference == 0 and document_1.rank == document_2.rank:
            return False
        elif preference > 0 and document_1.rank < document_2.rank:
            return False
        elif preference < 0 and document_1.rank > document_2.rank:
            return False
        else:
            return True

    @final
    def permutations(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[List[bool]]:
        return [
            [
                self.is_permutated(context, query, document1, document2)
                if index1 != index2 else False
                for index2, document2 in enumerate(ranking)
            ]
            for index1, document1 in enumerate(ranking)
        ]

    @final
    def permutation_count(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[int]:
        return [
            sum(1 for is_pair_permutated in pairs if is_pair_permutated)
            for pairs in self.permutations(context, query, ranking)
        ]

    @final
    def permutation_frequency(
            self,
            context: RerankingContext,
            query: Query,
            ranking: List[RankedDocument],
    ) -> List[float]:
        return [
            (count - 1) / len(ranking) if len(ranking) > 0 else 0
            for count in self.permutation_count(context, query, ranking)
        ]
