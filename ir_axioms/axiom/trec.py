from dataclasses import dataclass
from functools import lru_cache

from numpy import integer
from trectools import TrecQrel, TrecTopics

from ir_axioms import logger
from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


@dataclass(frozen=True)
class TrecOracleAxiom(Axiom):
    name = "trec-oracle"

    qrel: TrecQrel
    topics: TrecTopics

    @lru_cache
    def _topic(self, query: Query) -> int:
        topics = [
            topic
            for topic, topic_query in self.topics.topics.items()
            if topic_query == query.title
        ]
        if len(topics) == 0:
            raise RuntimeError(
                f"Could not find topic for query '{query.title}'."
            )
        elif len(topics) > 1:
            logger.warning(
                f"Found multiple topics for query '{query.title}': {topics}"
            )
        return topics[0]

    @lru_cache
    def _judgement(self, query: Query, document: RankedDocument) -> int:
        topic = self._topic(query)
        judgement = self.qrel.get_judgement(document.id, topic)
        if isinstance(judgement, integer):
            judgement = int(judgement)
        if not isinstance(judgement, int):
            raise RuntimeError(
                f"Invalid judgement for document {document.id} "
                f"in topic {topic}: {type(judgement)}"
            )
        return judgement

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        judgement1 = self._judgement(query, document1)
        judgement2 = self._judgement(query, document2)
        return strictly_greater(judgement1, judgement2)
