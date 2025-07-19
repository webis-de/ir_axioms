from dataclasses import dataclass
from math import isclose
from typing import Final, Union

from injector import inject

from ir_axioms.axiom.base import Axiom
from ir_axioms.dependency_injection import injector
from ir_axioms.model import Query, Document, ScoredDocument, Preference
from ir_axioms.tools import TextContents, TermTokenizer, TextStatistics
from ir_axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Lb1Axiom(Axiom[Query, ScoredDocument]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]
    margin_fraction: float = 0.1

    def preference(
        self,
        input: Query,
        output1: ScoredDocument,
        output2: ScoredDocument,
    ) -> Preference:
        if not isclose(
            output1.score,
            output2.score,
            rel_tol=self.margin_fraction,
        ):
            return 0

        query_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )

        document1_tf = self.text_statistics.term_frequencies(output1)
        document2_tf = self.text_statistics.term_frequencies(output2)

        query_term_only_in_document1 = any(
            document1_tf.get(term, 0) > 0 and document2_tf.get(term, 0) <= 0
            for term in query_terms
        )
        query_term_only_in_document2 = any(
            document2_tf.get(term, 0) > 0 and document1_tf.get(term, 0) <= 0
            for term in query_terms
        )

        if query_term_only_in_document1 and not query_term_only_in_document2:
            return 1
        elif not query_term_only_in_document1 and query_term_only_in_document2:
            return -1
        else:
            return 0


LB1: Final = lazy_inject(Lb1Axiom, injector)
