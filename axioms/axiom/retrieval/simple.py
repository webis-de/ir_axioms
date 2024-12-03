from dataclasses import dataclass

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less, strictly_greater
from axioms.model import Query, RankedDocument, IndexContext, JudgedRankedDocument


@dataclass(frozen=True, kw_only=True)
class OriginalAxiom(Axiom):

    def preference(
        self,
        context: IndexContext,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument,
    ):
        return strictly_less(document1.rank, document2.rank)


@dataclass(frozen=True, kw_only=True)
class OracleAxiom(Axiom):

    def preference(
        self,
        context: IndexContext,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument,
    ) -> float:
        if not isinstance(document1, JudgedRankedDocument) or not isinstance(
            document2, JudgedRankedDocument
        ):
            raise ValueError(
                f"Expected both documents to be "
                f"instances of {JudgedRankedDocument}, "
                f"but were {type(document1)} and {type(document2)}."
            )
        return strictly_greater(document1.relevance, document2.relevance)


# Aliases for shorter names:
ORIG = OriginalAxiom
ORACLE = OracleAxiom
