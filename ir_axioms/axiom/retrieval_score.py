from dataclasses import dataclass

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import strictly_greater
from ir_axioms.model import Query, RankedDocument
from ir_axioms.model.context import RerankingContext
from ir_axioms.model.retrieval_model import (
    RetrievalModel, Tf, TfIdf, BM25, DirichletLM, PL2
)


@dataclass(frozen=True)
class RetrievalScoreAxiom(Axiom):
    retrieval_model: RetrievalModel

    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        sd1 = context.retrieval_score(query, document1, self.retrieval_model)
        sd2 = context.retrieval_score(query, document2, self.retrieval_model)
        return strictly_greater(sd1, sd2)


class RS_TF(RetrievalScoreAxiom):
    name = "RS-TF"

    def __init__(self):
        super().__init__(Tf)


class RS_TF_IDF(RetrievalScoreAxiom):
    name = "RS-TF-IDF"

    def __init__(self):
        super().__init__(TfIdf)


class RS_BM25(RetrievalScoreAxiom):
    name = "RS-BM25"

    def __init__(self, k_1: float = 1.2, k_3: float = 8, b: float = 0.75):
        super().__init__(BM25(k_1, k_3, b))


class RS_PL2(RetrievalScoreAxiom):
    name = "RS-PL2"

    def __init__(self, c: float = 0.1):
        super().__init__(PL2(c))


class RS_QL(RetrievalScoreAxiom):
    name = "RS-QL"

    def __init__(self, mu: float = 1000):
        super().__init__(DirichletLM(mu))


# Aliases for shorter names:
RS = RetrievalScoreAxiom
