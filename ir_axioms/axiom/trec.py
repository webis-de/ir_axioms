from dataclasses import dataclass
from functools import lru_cache

from numpy import integer
from trectools import TrecQrel, TrecTopics

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, RankedDocument, IndexContext


@dataclass(frozen=True)
class TrecOracleAxiom(Axiom):
    name = "TREC-ORACLE"

    topics: TrecTopics
    qrels: TrecQrel

    @lru_cache(None)
    def _topic(self, query_title: str) -> int:
        topics = [
            topic
            for topic, topic_query in self.topics.topics.items()
            if topic_query == query_title
        ]
        if len(topics) == 0:
            raise RuntimeError(
                f"Could not find topic for query '{query_title}'."
            )
        if len(topics) > 1:
            raise RuntimeError(
                f"Found multiple topics for query '{query_title}': {topics}"
            )
        return topics[0]

    @lru_cache(None)
    def _judgement(self, query_title: str, document_id: str) -> int:
        topic = self._topic(query_title)
        judgement = self.qrels.get_judgement(document_id, topic)
        if isinstance(judgement, integer):
            judgement = int(judgement)
        if not isinstance(judgement, int):
            raise RuntimeError(
                f"Invalid judgement for document {document_id} "
                f"in topic {topic}: {type(judgement)}"
            )
        return judgement

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        judgement1 = self._judgement(query.title, document1.id)
        judgement2 = self._judgement(query.title, document2.id)
        return strictly_greater(judgement1, judgement2)


# Aliases for shorter names:
TREC = TrecOracleAxiom
