from dataclasses import dataclass
from functools import lru_cache

from numpy import integer
from trectools import TrecQrel, TrecTopics

from axioms.axiom.base import Axiom
from axioms.axiom.utils import strictly_greater
from axioms.model import Query, Document


@dataclass(frozen=True, kw_only=True)
class TrecOracleAxiom(Axiom[Query, Document]):
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
            raise RuntimeError(f"Could not find topic for query '{query_title}'.")
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
        input: Query,
        output1: Document,
        output2: Document,
    ) -> float:
        judgement1 = self._judgement(input.title, output1.id)
        judgement2 = self._judgement(input.title, output2.id)
        return strictly_greater(judgement1, judgement2)
