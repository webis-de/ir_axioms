from dataclasses import dataclass
from typing import Final, Union

from injector import inject

from axioms.axiom.base import Axiom
from axioms.dependency_injection import injector
from axioms.axiom.utils import approximately_equal, strictly_fewer, strictly_greater
from axioms.model import Query, Document, Preference
from axioms.tools import TextContents, TermTokenizer, TextStatistics
from axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Lnc1Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        if not all(
            approximately_equal(
                self.text_statistics.term_frequency(output1, term),
                self.text_statistics.term_frequency(output2, term),
            )
            for term in query_unique_terms
        ):
            return 0

        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )

        # Prefer the shorter document.
        return strictly_fewer(document1_terms, document2_terms)


LNC1: Final = lazy_inject(Lnc1Axiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class TfLncAxiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )

        sum_document1 = 0
        sum_document2 = 0

        for query_term in query_unique_terms:
            tf_d1 = self.text_statistics.term_frequency(output1, query_term)
            tf_d2 = self.text_statistics.term_frequency(output2, query_term)

            len_d1 = sum(1 for term in document1_terms if term != query_term)
            len_d2 = sum(1 for term in document2_terms if term != query_term)

            if len_d1 == len_d2:
                if tf_d1 > tf_d2:
                    sum_document1 += 1
                elif tf_d2 > tf_d1:
                    sum_document2 += 1

        return strictly_greater(sum_document1, sum_document2)


TF_LNC: Final = lazy_inject(TfLncAxiom, injector)
