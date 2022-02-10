from abc import abstractmethod, ABC
from typing import List

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import RankedDocument, Query, IndexContext


class EstimatorAxiom(Axiom, ABC):

    @abstractmethod
    def fit(
            self,
            context: IndexContext,
            query_documents: List[Query, List[RankedDocument]],
    ) -> str:
        pass
