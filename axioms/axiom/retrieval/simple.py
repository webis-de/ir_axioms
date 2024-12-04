from dataclasses import dataclass

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less, strictly_greater
from axioms.model import Query, RankedDocument, JudgedRankedDocument


@dataclass(frozen=True, kw_only=True)
class OriginalAxiom(Axiom[Query, RankedDocument]):

    def preference(
        self,
        input: Query,
        output1: RankedDocument,
        output2: RankedDocument,
    ):
        return strictly_less(output1.rank, output2.rank)


@dataclass(frozen=True, kw_only=True)
class OracleAxiom(Axiom[Query, JudgedRankedDocument]):

    def preference(
        self,
        input: Query,
        output1: JudgedRankedDocument,
        output2: JudgedRankedDocument,
    ) -> float:
        return strictly_greater(output1.relevance, output2.relevance)


# Aliases for shorter names:
ORIG = OriginalAxiom
ORACLE = OracleAxiom
