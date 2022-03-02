from dataclasses import dataclass
from math import isclose, prod, ceil
from typing import Iterable, Union

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Query, RankedDocument, IndexContext


@dataclass(frozen=True)
class UniformAxiom(Axiom):
    scalar: float

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return self.scalar


@dataclass(frozen=True)
class SumAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return sum(
            axiom.preference(context, query, document1, document2)
            for axiom in self.axioms
        )

    def __add__(self, other: Union[Axiom, float, int]) -> Axiom:
        if isinstance(other, Axiom):
            return SumAxiom([*self.axioms, other])
        else:
            return super().__add__(other)


@dataclass(frozen=True)
class ProductAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return prod(
            axiom.preference(context, query, document1, document2)
            for axiom in self.axioms
        )

    def __mul__(self, other: Union[Axiom, float, int]) -> Axiom:
        if isinstance(other, Axiom):
            # Avoid chaining operators.
            return ProductAxiom([*self.axioms, other])
        else:
            return super().__mul__(other)


@dataclass(frozen=True)
class MultiplicativeInverseAxiom(Axiom):
    axiom: Axiom

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return 1 / self.axiom.preference(
            context,
            query,
            document1,
            document2
        )


@dataclass(frozen=True)
class AndAxiom(Axiom):
    # TODO: And is a special case of majority vote with a majority of 1.0
    #   We might want to merge both classes eventually.
    axioms: Iterable[Axiom]

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preferences = [
            axiom.preference(context, query, document1, document2)
            for axiom in self.axioms
        ]
        if all(preference > 0 for preference in preferences):
            return 1
        elif all(preference > 0 for preference in preferences):
            return -1
        else:
            return 0

    def __and__(self, other: Union[Axiom, float, int]) -> Axiom:
        if isinstance(other, Axiom):
            # Avoid chaining operators.
            return AndAxiom([*self.axioms, other])
        else:
            return super().__and__(other)


@dataclass(frozen=True)
class VoteAxiom(Axiom):
    axioms: Iterable[Axiom]
    minimum_votes: float = 0.5
    """
    Minimum portion of votes in favor or against either document,
    to be considered a majority,
    for example, 0.5 for absolute majority, 0.6 for qualified majority,
    or 0 for relative majority.
    """

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        axioms = tuple(self.axioms)
        preferences = (
            axiom.preference(context, query, document1, document2)
            for axiom in axioms
        )

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

        if (
                positive_votes > negative_votes and
                positive_votes >= minimum_votes
        ):
            return 1
        elif (
                negative_votes > positive_votes and
                negative_votes >= minimum_votes
        ):
            return -1
        else:
            # Draw.
            return 0

    def __mod__(self, other: Union[Axiom, float, int]) -> Axiom:
        if isinstance(other, Axiom) and isclose(self.minimum_votes, 0.5):
            # Avoid chaining operators
            # if this vote has the default minimum vote proportion.
            return VoteAxiom([*self.axioms, other])
        else:
            return super().__mod__(other)


@dataclass(frozen=True)
class CascadeAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preferences = (
            axiom.preference(context, query, document1, document2)
            for axiom in self.axioms
        )
        decisive_preferences = (
            preference
            for preference in preferences
            if preference != 0
        )
        return next(decisive_preferences, 0)

    def __or__(self, other: Union[Axiom, float, int]) -> Axiom:
        if isinstance(other, Axiom):
            # Avoid chaining operators.
            return CascadeAxiom([*self.axioms, other])
        else:
            return super().__or__(other)


@dataclass(frozen=True)
class NormalizedAxiom(Axiom):
    axiom: Axiom

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preference = self.axiom.preference(
            context,
            query,
            document1,
            document2
        )
        if preference > 0:
            return 1
        elif preference < 0:
            return -1
        else:
            return 0

    def __pos__(self) -> Axiom:
        # This axiom is already normalized.
        return self
