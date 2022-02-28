from dataclasses import dataclass

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, RankedDocument, IndexContext
from ir_axioms.model.retrieval_model import (
    RetrievalModel, Tf, TfIdf, BM25, PL2, QL
)


@dataclass(frozen=True)
class RetrievalScoreAxiom(Axiom):
    name = "RetrievalScore"

    retrieval_model: RetrievalModel

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_greater(
            context.retrieval_score(query, document1, self.retrieval_model),
            context.retrieval_score(query, document2, self.retrieval_model),
        )


@dataclass(frozen=True)
class RS_TF(Axiom, Tf):
    name = "RS-TF"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_greater(
            context.retrieval_score(query, document1, self),
            context.retrieval_score(query, document2, self),
        )


@dataclass(frozen=True)
class RS_TF_IDF(Axiom, TfIdf):
    name = "RS-TF-IDF"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_greater(
            context.retrieval_score(query, document1, self),
            context.retrieval_score(query, document2, self),
        )


@dataclass(frozen=True)
class RS_BM25(Axiom, BM25):
    name = "RS-BM25"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_greater(
            context.retrieval_score(query, document1, self),
            context.retrieval_score(query, document2, self),
        )


@dataclass(frozen=True)
class RS_PL2(Axiom, PL2):
    name = "RS-PL2"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_greater(
            context.retrieval_score(query, document1, self),
            context.retrieval_score(query, document2, self),
        )


@dataclass(frozen=True)
class RS_QL(Axiom, QL):
    name = "RS-QL"

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        return strictly_greater(
            context.retrieval_score(query, document1, self),
            context.retrieval_score(query, document2, self),
        )


# Aliases for shorter names:
RS = RetrievalScoreAxiom
