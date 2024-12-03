from dataclasses import dataclass
from functools import reduce
from math import isclose, ceil
from operator import add, mul
from typing import Any, Iterable, Sequence

from numpy import full, ones, zeros
from typing_extensions import TypeAlias  # type: ignore

from axioms.axiom.base import Axiom
from axioms.model import Input, Output, Preference, PreferenceMatrix


@dataclass(frozen=True, kw_only=True)
class UniformAxiom(Axiom[Any, Any]):
    scalar: float

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return self.scalar

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        return full((len(outputs), len(outputs)), self.scalar)


@dataclass(frozen=True, kw_only=True)
class SumAxiom(Axiom[Input, Output]):
    axioms: Iterable[Axiom[Input, Output]]

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return sum(axiom.preference(input, output1, output2) for axiom in self.axioms)

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        return reduce(
            add,
            (axiom.preferences(input, outputs) for axiom in self.axioms),
            zeros((len(outputs), len(outputs))),
        )

    def __add__(self, other: Axiom[Input, Output]) -> Axiom[Input, Output]:
        return SumAxiom(axioms=[*self.axioms, other])


@dataclass(frozen=True, kw_only=True)
class ProductAxiom(Axiom[Input, Output]):
    axioms: Iterable[Axiom[Input, Output]]

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return reduce(
            mul, (axiom.preference(input, output1, output2) for axiom in self.axioms), 1
        )

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        return reduce(
            mul,
            (axiom.preferences(input, outputs) for axiom in self.axioms),
            ones((len(outputs), len(outputs))),
        )

    def __mul__(self, other: Axiom[Input, Output]) -> Axiom[Input, Output]:
        # Avoid chaining operators.
        return ProductAxiom(axioms=[*self.axioms, other])


@dataclass(frozen=True, kw_only=True)
class MultiplicativeInverseAxiom(Axiom[Input, Output]):
    axiom: Axiom[Input, Output]

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return 1 / self.axiom.preference(input, output1, output2)

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        return 1 / self.axiom.preferences(input, outputs)

    def __rtruediv__(self, other: Axiom[Input, Output]) -> Axiom[Input, Output]:
        # Avoid chaining operators.
        return self.axiom * other


@dataclass(frozen=True, kw_only=True)
class ConjunctionAxiom(Axiom[Input, Output]):
    # TODO: And is a special case of majority vote with a majority of 1.0
    #   We might want to merge both classes eventually.
    axioms: Iterable[Axiom[Input, Output]]

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        preferences = [
            axiom.preference(input, output1, output2) for axiom in self.axioms
        ]
        if all(preference > 0 for preference in preferences):
            return 1
        elif all(preference > 0 for preference in preferences):
            return -1
        else:
            return 0

    # TODO: Add batched preference computation.

    def __and__(self, other: Axiom[Input, Output]) -> Axiom[Input, Output]:
        # Avoid chaining operators.
        return ConjunctionAxiom(axioms=[*self.axioms, other])


@dataclass(frozen=True, kw_only=True)
class VoteAxiom(Axiom[Input, Output]):
    axioms: Iterable[Axiom[Input, Output]]
    minimum_votes: float = 0.5
    """
    Minimum portion of votes in favor or against either document,
    to be considered a majority,
    for example, 0.5 for absolute majority, 0.6 for qualified majority,
    0 for relative majority, or 1 for consensus.
    """

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        axioms = tuple(self.axioms)
        preferences = (axiom.preference(input, output1, output2) for axiom in axioms)

        # Total count of possible votes.
        count: int = len(axioms)

        # Minimum (absolute) number of votes to reach a majority.
        minimum_votes: int = ceil(self.minimum_votes * count)

        # Number of observed positive votes.
        positive_votes: int = 0
        # Number of observed negative votes.
        negative_votes: int = 0
        # Number of observed neutral votes.
        neutral_votes: int = 0

        for preference in preferences:
            if preference > 0:
                positive_votes += 1
            elif preference < 0:
                negative_votes += 1
            else:
                neutral_votes += 1
        # TODO: Optimize by comparing majorities with "open" votes.

        if positive_votes > negative_votes and positive_votes >= minimum_votes:
            return 1
        elif negative_votes > positive_votes and negative_votes >= minimum_votes:
            return -1
        else:
            # Draw.
            return 0

    # TODO: Add batched preference computation.

    def __mod__(self, other: Axiom[Input, Output]) -> Axiom[Input, Output]:
        if isclose(self.minimum_votes, 0.5):
            # Avoid chaining operators
            # if this vote has the default minimum vote proportion.
            return VoteAxiom(axioms=[*self.axioms, other])
        else:
            return super().__mod__(other)


class MajorityVoteAxiom(VoteAxiom):
    pass


@dataclass(frozen=True, kw_only=True)
class CascadeAxiom(Axiom[Input, Output]):
    axioms: Iterable[Axiom[Input, Output]]

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        preferences = (
            axiom.preference(input, output1, output2) for axiom in self.axioms
        )
        decisive_preferences = (
            preference for preference in preferences if preference != 0
        )
        return next(decisive_preferences, 0)

    # TODO: Add batched preference computation.

    def __or__(self, other: Axiom[Input, Output]) -> Axiom[Input, Output]:
        # Avoid chaining operators.
        return CascadeAxiom(axioms=[*self.axioms, other])


@dataclass(frozen=True, kw_only=True)
class NormalizedAxiom(Axiom[Input, Output]):
    axiom: Axiom[Input, Output]

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        preference = self.axiom.preference(input, output1, output2)
        if preference > 0:
            return 1
        elif preference < 0:
            return -1
        else:
            return 0

    # TODO: Add batched preference computation.

    def __pos__(self) -> Axiom[Input, Output]:
        # This axiom is already normalized.
        return self
