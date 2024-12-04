from abc import ABC, abstractmethod
from typing import Generic, Literal, Sequence, Optional, final

from numpy import floating, array

from axioms.model import Input, Output, Preference, PreferenceMatrix
from axioms.tools.pivot import PivotSelection, RandomPivotSelection
from axioms.precondition.base import Precondition


class Axiom(ABC, Generic[Input, Output]):
    """
    An axiom describes a pairwise constraint between two outputs given the same input, as expressed as a preference.

    Subclasses must implement the ``preference()`` method that determines the pairwise preference between two outputs, and can optionally also override ``preference_matrix()`` to batch-compute a full preference matrix of arbitrarily many outputs more efficiently.

    This base class also exposes various operators (i.e., ``+``, ``-``, ``*``, ``/``, ``%``, ``&``, ``~``) for combining and manipulating axioms, as well as ``rerank()`` for KwikSort re-ranking the outputs, and other methods for evaluating rankings of outputs in comparison to the axiom's preferences.
    """

    @abstractmethod
    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        """
        Compute the pairwise preference between two outputs for the same input.
        The function should return a positive value if `output1` is to be prefered, a negative value if `output2` is to be prefered, and zero if there is no preference or if some precondition was not met.

        Note that the preference should be independent of the order of ``output1`` and ``output2``, i.e., if the inputs are flipped, the output should be flipped as well.

        :param input: Common input for both outputs.
        :param output1: One output for the common input.
        :param output2: Another output for the common input.
        :return: >0 if ``output1`` should be preferred,
        <0 if ``output2`` should be preferred,
        or 0 if neither of the outputs should be preferred over the other.
        """
        pass

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        """
        Batch-compute the preferences for a sequence of potential outputs.
        While the naive default implementation just delegates to `preference()`, it might make sense to override it to avoid computating certain costly features again and again for the same output.

        :param input: Common input for all outputs.
        :param outputs: The outputs for the common input.
        :return: A preference matrix, where the ij-th entry corresponds to the preference between the i-th and j-th output.
        """

        return array(
            [
                [
                    self.preference(
                        input=input,
                        output1=output1,
                        output2=output2,
                    )
                    for output2 in outputs
                ]
                for output1 in outputs
            ],
            dtype=floating,
        )

    def __add__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import SumAxiom

        return SumAxiom(axioms=[self, other])

    def __radd__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return self + other

    def __sub__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return self + -other

    def __rsub__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return -self + other

    def __mul__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import ProductAxiom

        return ProductAxiom(axioms=[self, other])

    def __rmul__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return self * other

    def __truediv__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import MultiplicativeInverseAxiom

        return self * MultiplicativeInverseAxiom(axiom=other)

    def __rtruediv__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import MultiplicativeInverseAxiom

        return MultiplicativeInverseAxiom(axiom=self) * other

    def __mod__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import VoteAxiom

        return VoteAxiom(axioms=[self, other])

    def __rmod__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return self % other

    def majority_vote(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return self % other

    def __and__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import ConjunctionAxiom

        return ConjunctionAxiom(axioms=[self, other])

    def __rand__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return self & other

    def __or__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import CascadeAxiom

        return CascadeAxiom(axioms=[self, other])

    def __ror__(self, other: "Axiom"[Input, Output]) -> "Axiom"[Input, Output]:
        return other | self

    def __neg__(self) -> "Axiom"[Input, Output]:
        from axioms.axiom.arithmetic import UniformAxiom

        return self * UniformAxiom(scalar=-1)

    def __pos__(self) -> "Axiom"[Input, Output]:
        """
        Return the normalized preference of this axiom,
        replacing positive values with 1 and negative values with -1.
        """
        from axioms.axiom.arithmetic import NormalizedAxiom

        return NormalizedAxiom(axiom=self)

    def normalized(self) -> "Axiom"[Input, Output]:
        """
        Return the normalized preference of this axiom,
        replacing positive values with 1 and negative values with -1.
        """
        return +self

    def __invert__(self) -> "Axiom"[Input, Output]:
        """
        Cache this axiom's preferences in the context's cache directory,
        meaning the ``preference()`` method will only be called once
        for each query-documents tuple.
        """
        return self.cached()

    def cached(self) -> "Axiom"[Input, Output]:
        """
        Cache this axiom's preferences in the context's cache directory,
        meaning the ``preference()`` method will only be called once
        for each query-documents tuple.
        """
        from axioms.axiom.cache import CachedAxiom

        return CachedAxiom(axiom=self)

    def parallel(self, n_jobs: Optional[int] = None) -> "Axiom"[Input, Output]:
        """
        Parallelize preference matrix computation of this axiom.
        """
        from axioms.axiom.parallel import ParallelAxiom

        return ParallelAxiom(axiom=self, n_jobs=n_jobs)

    def as_precondition(
        self,
        expected_sign: Literal[1, 0, -1] = 0,
        strip_preconditions: bool = True,
    ) -> Precondition[Input, Output]:
        from axioms.precondition.axiom import AxiomPrecondition

        return AxiomPrecondition(
            axiom=self,
            expected_sign=expected_sign,
            strip_preconditions=strip_preconditions,
        )

    def with_precondition(
        self,
        precondition: Precondition[Input, Output],
    ) -> "Axiom"[Input, Output]:
        from axioms.axiom.precondition import PreconditionAxiom

        return PreconditionAxiom(
            axiom=self,
            precondition=precondition,
        )

    @final
    def rerank_kwiksort(
        self,
        input: Input,
        ranking: Sequence[Output],
        pivot_selection: PivotSelection[Input, Output] = RandomPivotSelection(),
    ) -> Sequence[Output]:
        from axioms.algorithms.ranking import kwiksort

        ranking = kwiksort(
            axiom=self,
            input=input,
            vertices=ranking,
            pivot_selection=pivot_selection,
        )
        return ranking
