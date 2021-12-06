from abc import abstractmethod, ABC
from dataclasses import dataclass

from ir_axioms.axiom import Axiom
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext


class _RetrievalScoreAxiom(Axiom, ABC):
    @abstractmethod
    def retrieval_score(
            self,
            context: RerankingContext,
            query: Query,
            document: RankedDocument
    ) -> float:
        pass

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        sd1 = self.retrieval_score(context, query, document1)
        sd2 = self.retrieval_score(context, query, document2)
        return 1 if sd1 > sd2 else (-1 if sd1 < sd2 else 0)


class RS_TF(_RetrievalScoreAxiom):
    def retrieval_score(
            self,
            context: RerankingContext,
            query: Query,
            document: RankedDocument
    ) -> float:
        length = len(context.terms(document.content))
        return sum(
            context.term_frequency(document.content, term)
            for term in context.terms(query.title)
        ) / length


class RS_TF_IDF(_RetrievalScoreAxiom):
    def retrieval_score(
            self,
            context: RerankingContext,
            query: Query,
            document: RankedDocument
    ) -> float:
        return context.tf_idf_score(query, document)


@dataclass(frozen=True)
class RS_BM25(_RetrievalScoreAxiom):
    k1: float = 1.2
    b: float = 0.75

    def retrieval_score(
            self,
            context: RerankingContext,
            query: Query,
            document: RankedDocument
    ) -> float:
        return context.bm25_score(query, document, self.k1, self.b)


@dataclass(frozen=True)
class RS_PL2(_RetrievalScoreAxiom):
    c: float = 0.1

    def retrieval_score(
            self,
            context: RerankingContext,
            query: Query,
            document: RankedDocument
    ) -> float:
        return context.pl2_score(query, document, self.c)


@dataclass(frozen=True)
class RS_QL(_RetrievalScoreAxiom):
    mu: float = 1000

    def retrieval_score(
            self,
            context: RerankingContext,
            query: Query,
            document: RankedDocument
    ) -> float:
        return context.ql_score(query, document, self.mu)
