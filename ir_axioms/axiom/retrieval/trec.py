from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

from numpy import array, float_, integer
from trectools import TrecQrel

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, Document, PreferenceMatrix


@dataclass(frozen=True, kw_only=True)
class TrecOracleAxiom(Axiom[Query, Document]):
    qrels: TrecQrel

    @cached_property
    def _judgements(self) -> dict[tuple[str, str], int]:
        return {}

    def _judgement(self, query_id: str, document_id: str) -> int:
        key = (query_id, document_id)
        if key in self._judgements:
            return self._judgements[key]
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
        self._judgements[key] = judgement
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

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        judgements = [self._judgement(input.id, output.id) for output in outputs]
        return array(
            [
                strictly_greater(judgement1, judgement2)
                for judgement1 in judgements
                for judgement2 in judgements
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))
