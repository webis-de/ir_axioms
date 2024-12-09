from dataclasses import dataclass
from functools import lru_cache

from numpy import integer
from trectools import TrecQrel

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_greater
from axioms.model import Query, Document


@dataclass(frozen=True, kw_only=True)
class TrecOracleAxiom(Axiom[Query, Document]):
    qrels: TrecQrel

    @lru_cache(None)
    def _judgement(self, query_id: str, document_id: str) -> int:
        judgement = self.qrels.get_judgement(
            document=document_id,
            topic=query_id,
        )
        if isinstance(judgement, integer):
            judgement = int(judgement)
        if not isinstance(judgement, int):
            raise RuntimeError(
                f"Invalid judgement for document {document_id} "
                f"in topic {query_id}: {type(judgement)}"
            )
        return judgement

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> float:
        judgement1 = self._judgement(input.id, output1.id)
        judgement2 = self._judgement(input.id, output2.id)
        return strictly_greater(judgement1, judgement2)
