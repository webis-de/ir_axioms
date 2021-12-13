from dataclasses import dataclass, field
from typing import Union, List, Collection

from nltk import word_tokenize

from ir_axioms.model import Query, Document
from ir_axioms.model.context import RerankingContext
from ir_axioms.model.retrieval_model import RetrievalModel
from ir_axioms.utils import text_content


@dataclass(frozen=True)
class MemoryRerankingContext(RerankingContext):
    documents: Collection[Document]

    def __hash__(self):
        return sum(hash(document) for document in self.documents)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def document_frequency(self, term: str) -> int:
        print(
            f"{term}: "
            f"{[(d.id, term in self.terms(d)) for d in self.documents]} = "
            f"{sum(1 for d in self.documents if term in self.terms(d))}"
        )
        return sum(
            1
            for document in self.documents
            if term in self.terms(document)
        )

    def terms(self, query_or_document: Union[Query, Document]) -> List[str]:
        text = text_content(query_or_document)
        return word_tokenize(text)

    def retrieval_score(
            self,
            query: Query,
            document: Document,
            model: RetrievalModel
    ) -> float:
        raise NotImplementedError()
