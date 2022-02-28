from abc import ABC, abstractmethod
from inspect import isabstract
from math import nan
from typing import final, Union, Callable, Sequence, Optional

from numpy import ndarray, array, stack

from ir_axioms import registry
from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.modules.pivot import PivotSelection, RandomPivotSelection


class Axiom(ABC):
    """
    Base class for all axioms.
    Implements the various operators
    ``+``, ``-``, ``*``, ``/``, ``%``, ``&``, ``~``
    as well as ``rerank()`` for re-ranking with KwikSort,
    ``preferences()`` for collecting all preferences for a ranking,
    and other methods for evaluating rankings
    in comparison to the axiom's preferences.
    """

    name: str
    """
    The axiom classes unique, short name, describing its behavior.
    """

    def __init_subclass__(cls, **kwargs):
        # Automatically register this subclass to the global axiom registry.
        if (
                not isabstract(cls) and
                hasattr(cls, "name") and
                cls.name is not NotImplemented
        ):
            registry[cls.name] = cls

    @abstractmethod
    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        """
        Return whether to prefer the first document (return value > 0),
        the second document (return value < 0), or neither (return value = 0),
        when retrieving a query, given a reranking context.

        Note that the order of ``document1`` and ``document2``
        in the original ranking is *only* determined
        by their ``rank`` attributes,
        not by their order in the ``preference()`` function invocation.

        :param context: Reranking context for accessing index statistics
        and retrieval scores.
        :param query: Query for the original ranking
        :param document1: Document from an original ranking.
        :param document2: Document from an original ranking.
        :return: >0 if ``document1`` should be preferred,
        <0 if ``document2`` should be preferred,
        or 0 if neither document should be preferred over the other.
        """
        pass

    def __add__(self, other: "AxiomLike") -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import SumAxiom
            return SumAxiom([self, other])
        elif isinstance(other, (float, int, str)):
            from ir_axioms.axiom.conversion import to_axiom
            return self + to_axiom(other)
        else:
            return NotImplemented

    def __radd__(self, other: "AxiomLike") -> "Axiom":
        return self + other

    def plus(self, other: "AxiomLike") -> "Axiom":
        return self + other

    def __sub__(self, other: "AxiomLike") -> "Axiom":
        return self + -other

    def __rsub__(self, other: "AxiomLike") -> "Axiom":
        return -self + other

    def minus(self, other: "AxiomLike") -> "Axiom":
        return self - other

    def __mul__(self, other: "AxiomLike") -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import ProductAxiom
            return ProductAxiom([self, other])
        elif isinstance(other, (float, int, str)):
            from ir_axioms.axiom.conversion import to_axiom
            return self * to_axiom(other)
        else:
            return NotImplemented

    def __rmul__(self, other: "AxiomLike") -> "Axiom":
        return self * other

    def times(self, other: "AxiomLike") -> "Axiom":
        return self * other

    def weighted(self, weight: float) -> "Axiom":
        return self * weight

    def __truediv__(self, other: "AxiomLike") -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import MultiplicativeInverseAxiom
            return self * MultiplicativeInverseAxiom(other)
        elif isinstance(other, (float, int, str)):
            from ir_axioms.axiom.conversion import to_axiom
            return self / to_axiom(other)
        else:
            return NotImplemented

    def __rtruediv__(self, other: "AxiomLike") -> "Axiom":
        from ir_axioms.axiom.arithmetic import MultiplicativeInverseAxiom
        return MultiplicativeInverseAxiom(self) * other

    def divide(self, other: "AxiomLike") -> "Axiom":
        return self / other

    def __mod__(self, other: "AxiomLike") -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import VoteAxiom
            return VoteAxiom([self, other])
        elif isinstance(other, (float, int, str)):
            from ir_axioms.axiom.conversion import to_axiom
            return self % to_axiom(other)
        else:
            return NotImplemented

    def __rmod__(self, other: "AxiomLike") -> "Axiom":
        return self % other

    def majority_vote(self, other: "AxiomLike") -> "Axiom":
        return self % other

    def __and__(self, other: "AxiomLike") -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import AndAxiom
            return AndAxiom([self, other])
        elif isinstance(other, (float, int, str)):
            from ir_axioms.axiom.conversion import to_axiom
            return self & to_axiom(other)
        else:
            return NotImplemented

    def __rand__(self, other: "AxiomLike") -> "Axiom":
        return self & other

    def __or__(self, other: "AxiomLike") -> "Axiom":
        if isinstance(other, Axiom):
            from ir_axioms.axiom.arithmetic import CascadeAxiom
            return CascadeAxiom([self, other])
        elif isinstance(other, (float, int, str)):
            from ir_axioms.axiom.conversion import to_axiom
            return self | to_axiom(other)
        else:
            return NotImplemented

    def __ror__(self, other: "AxiomLike") -> "Axiom":
        if isinstance(other, Axiom):
            return other | self
        elif isinstance(other, (float, int, str)):
            from ir_axioms.axiom.conversion import to_axiom
            return to_axiom(other) | self
        else:
            return NotImplemented

    def __neg__(self) -> "Axiom":
        return self * -1

    def __pos__(self) -> "Axiom":
        """
        Return the normalized preference of this axiom,
        replacing positive values with 1 and negative values with -1.
        """
        from ir_axioms.axiom.arithmetic import NormalizedAxiom
        return NormalizedAxiom(self)

    def normalized(self) -> "Axiom":
        """
        Return the normalized preference of this axiom,
        replacing positive values with 1 and negative values with -1.
        """
        return +self

    def __invert__(self) -> "Axiom":
        """
        Cache this axiom's preferences in the context's cache directory,
        meaning the ``preference()`` method will only be called once
        for each query-documents tuple.
        """
        return self.cached()

    def cached(self) -> "Axiom":
        """
        Cache this axiom's preferences in the context's cache directory,
        meaning the ``preference()`` method will only be called once
        for each query-documents tuple.
        """
        from ir_axioms.axiom.cache import CachedAxiom
        return CachedAxiom(self)

    @final
    def rerank_kwiksort(
            self,
            context: IndexContext,
            query: Query,
            ranking: Sequence[RankedDocument],
            pivot_selection: PivotSelection = RandomPivotSelection(),
    ) -> Sequence[RankedDocument]:
        from ir_axioms.modules.ranking import kwiksort, reset_score

        ranking = kwiksort(self, query, context, ranking, pivot_selection)
        ranking = reset_score(ranking)
        return ranking

    @final
    def preference_matrix(
            self,
            context: IndexContext,
            query: Query,
            ranking: Sequence[RankedDocument],
            filter_pairs: Optional[Callable[
                [RankedDocument, RankedDocument],
                bool
            ]] = None,
    ) -> ndarray:
        return stack(tuple(
            array(tuple(
                self.preference(context, query, document1, document2)
                if filter_pairs is None or filter_pairs(document1, document2)
                else nan
                for document2 in ranking
            ))
            for document1 in ranking
        ))


AxiomLike = Union[Axiom, str, int, float]
