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

    qrels: DataFrame
    topics: DataFrame

    @lru_cache
    def _topic_id(self, query: Query) -> Optional[int]:
        topics = self.topics[self.topics["query"] == query.title]
        if len(topics) == 0:
            return None
        elif len(topics) > 1:
            logger.warning(
                f"Found multiple topics for query '{query.title}': {topics['qid'].to_list()}"
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
        qrels = self.qrels[
            self.qrels["qid"] == topic_id &
            self.qrels["docno"] == document.id
            ]
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
