from dataclasses import dataclass
from typing import Any, Final

from axioms.axiom.utils import approximately_equal
from axioms.model import Document, IndexContext, Mask
from axioms.model.retrieval import get_index_context
from axioms.precondition.base import Precondition


@dataclass(frozen=True, kw_only=True)
class LenPrecondition(Precondition[Any, Document]):
    context: IndexContext
    margin_fraction: float = 0.1

    def precondition(
        self,
        input: Any,
        output1: Document,
        output2: Document,
    ) -> Mask:
        return approximately_equal(
            self.context.document_length(output1),
            self.context.document_length(output2),
            margin_fraction=self.margin_fraction,
        )


LEN: Final = LenPrecondition(
    context=get_index_context(),
)
