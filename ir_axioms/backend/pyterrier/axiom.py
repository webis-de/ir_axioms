from dataclasses import dataclass
from functools import lru_cache, cached_property
from math import nan
from typing import Optional

from pandas import DataFrame

from ir_axioms import logger
from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class OracleAxiom(Axiom):
    name = "oracle"

    _qrels_topics: DataFrame

    def __init__(
            self,
            topics: DataFrame,
            qrels: DataFrame,
    ):
        assert "query" in topics.columns
        assert "qid" in topics.columns
        assert "qid" in qrels.columns
        assert "docno" in qrels.columns
        assert "label" in qrels.columns

        self._qrels_topics = topics.merge(qrels, on=["qid"])
        del self._qrels_topics["qid"]

    @cached_property
    def _qrels_topics_hash(self) -> int:
        return hash(self._qrels_topics.to_json())

    def __hash__(self):
        return self._qrels_topics_hash

    @lru_cache
    def _judgement(
            self,
            query: Query,
            document: RankedDocument
    ) -> Optional[int]:
        qrels = self._qrels_topics
        qrels = qrels[qrels["query"] == query.title]
        qrels = qrels[qrels["docno"] == document.id]
        if len(qrels) == 0:
            return None
        elif len(qrels) > 1:
            logger.warning(
                f"Found multiple qrels for topic '{query.title}', "
                f"document {document.id}: {qrels['label'].to_list()}"
            )
        return qrels.iloc[0]["label"]

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        judgement1 = self._judgement(query, document1)
        judgement2 = self._judgement(query, document2)
        if judgement1 is None or judgement2 is None:
            return nan
        return strictly_greater(judgement1, judgement2)
