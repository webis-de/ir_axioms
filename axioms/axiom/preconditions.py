from dataclasses import dataclass
from inspect import isabstract

from axioms.axiom.base import Axiom
from axioms.axiom.utils import approximately_equal
from axioms.model import IndexContext, Query, RankedDocument


def approximately_same_length(
        context: IndexContext,
        document1: RankedDocument,
        document2: RankedDocument,
        margin_fraction: float,
) -> bool:
    return approximately_equal(
        context.document_length(document1),
        context.document_length(document2),
        margin_fraction=margin_fraction
    )


@dataclass(frozen=True, kw_only=True)
class LEN(Axiom):
    name = "LEN"

    axiom: Axiom
    margin_fraction: float = 0.1

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not approximately_same_length(
                context,
                document1,
                document2,
                self.margin_fraction,
        ):
            # Documents have different lengths.
            return 0

        return self.axiom.preference(context, query, document1, document2)


@dataclass(frozen=True, kw_only=True)
class LEN_Mixin(Axiom):
    margin_fraction: float = 0.1

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not approximately_same_length(
                context,
                document1,
                document2,
                self.margin_fraction,
        ):
            # Documents have different lengths.
            return 0
        if isabstract(super()):
            raise RuntimeError("This class must be used as a mixin.")
        return super().preference(context, query, document1, document2)  # type: ignore
