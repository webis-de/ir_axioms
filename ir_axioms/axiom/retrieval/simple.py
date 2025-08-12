from dataclasses import dataclass
from typing import Any

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_less, strictly_greater
from ir_axioms.model import Document
from ir_axioms.utils.lazy import lazy_inject


@dataclass(frozen=True, kw_only=True)
class OriginalAxiom(Axiom[Any, Document]):
    def preference(
        self,
        input: Any,
        output1: Document,
        output2: Document,
    ) -> float:
        if output1.score is not None and output2.score is not None:
            return strictly_greater(output1.score, output2.score)
        elif output1.rank is not None and output2.rank is not None:
            return strictly_less(output1.rank, output2.rank)
        else:
            raise ValueError("Can only compare ranked or scored documents.")


ORIG = lazy_inject(OriginalAxiom)


@dataclass(frozen=True, kw_only=True)
class OracleAxiom(Axiom[Any, Document]):
    def preference(
        self,
        input: Any,
        output1: Document,
        output2: Document,
    ) -> float:
        if output1.relevance is not None and output2.relevance is not None:
            return strictly_greater(output1.relevance, output2.relevance)
        else:
            raise ValueError("Can only compare judged documents.")


ORACLE = lazy_inject(OracleAxiom)
