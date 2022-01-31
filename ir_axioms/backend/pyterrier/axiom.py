from dataclasses import dataclass
from functools import lru_cache
from math import nan
from typing import Optional

from pandas import DataFrame

from ir_axioms import logger
from ir_axioms.axiom import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


@dataclass(frozen=True)
class OracleAxiom(Axiom):
    name = "oracle"

    topics: DataFrame
    qrels: DataFrame

    def __post_init__(self):
        assert "query" in self.topics.columns
        assert "qid" in self.topics.columns
        assert "qid" in self.qrels.columns
        assert "docno" in self.qrels.columns
        assert "label" in self.qrels.columns

    def __hash__(self):
        return hash((self.qrels.to_json(), self.topics.to_json()))

    @lru_cache
    def _topic_id(self, query: Query) -> Optional[int]:
        topics = self.topics
        topics = topics[topics["query"] == query.title]
        if len(topics) == 0:
            return None
        elif len(topics) > 1:
            logger.warning(
                f"Found multiple topics for query '{query.title}': "
                f"{topics['qid'].to_list()}"
            )
        return topics.iloc[0]["qid"]

    @lru_cache
    def _judgement(
            self,
            query: Query,
            document: RankedDocument
    ) -> Optional[int]:
        topic_id = self._topic_id(query)
        if topic_id is None:
            return None
        qrels = self.qrels
        qrels = qrels[qrels["qid"] == topic_id]
        qrels = qrels[qrels["docno"] == document.id]
        if len(qrels) == 0:
            return None
        elif len(qrels) > 1:
            logger.warning(
                f"Found multiple qrels for topic {topic_id}, "
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
