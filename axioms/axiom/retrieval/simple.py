from dataclasses import dataclass
from typing import Any, TypeVar

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_less, strictly_greater
from axioms.dependency_injection import injector
from axioms.model import ScoredDocument, RankedDocument, JudgedDocument
from axioms.utils.lazy import lazy_inject


_RankedOrScoredDocument = TypeVar(
    "_RankedOrScoredDocument", RankedDocument, ScoredDocument
)


@dataclass(frozen=True, kw_only=True)
class OriginalAxiom(Axiom[Any, _RankedOrScoredDocument]):

    def preference(
        self,
        input: Any,
        output1: _RankedOrScoredDocument,
        output2: _RankedOrScoredDocument,
    ) -> float:
        if isinstance(output1, ScoredDocument) and isinstance(output2, ScoredDocument):
            return strictly_greater(output1.score, output2.score)
        elif isinstance(output1, RankedDocument) and isinstance(
            output2, RankedDocument
        ):
            return strictly_less(output1.rank, output2.rank)
        else:
            raise ValueError("Can only compare RankedDocument's or ScoredDocument's.")


ORIG = lazy_inject(OriginalAxiom, injector)


@dataclass(frozen=True, kw_only=True)
class OracleAxiom(Axiom[Any, JudgedDocument]):

    def preference(
        self,
        input: Any,
        output1: JudgedDocument,
        output2: JudgedDocument,
    ) -> float:
        return strictly_greater(output1.relevance, output2.relevance)


ORACLE = lazy_inject(OracleAxiom, injector)
