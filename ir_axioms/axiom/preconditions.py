from dataclasses import dataclass

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.utils import approximately_equal
from ir_axioms.model import IndexContext, Query, RankedDocument


def approximately_same_length(
        context: IndexContext,
        document1: RankedDocument,
        document2: RankedDocument,
        margin_fraction: float,
) -> bool:
    return approximately_equal(
        len(context.terms(document1)),
        len(context.terms(document2)),
        margin_fraction=margin_fraction
    )


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
        return super().preference(context, query, document1, document2)
