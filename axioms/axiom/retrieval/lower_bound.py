from dataclasses import dataclass
from math import isclose
from typing import Final, Union

from injector import inject

from axioms.axiom.base import Axiom
from axioms.axiom.utils import approximately_equal
from axioms.dependency_injection import injector
from axioms.model import Query, Document, ScoredDocument, Preference
from axioms.tools import TextContents, TermTokenizer, TextStatistics
from axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Lb1Axiom(Axiom[Query, ScoredDocument]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]

    def preference(
        self,
        input: Query,
        output1: ScoredDocument,
        output2: ScoredDocument,
    ) -> Preference:
        if not approximately_equal(output1.score, output2.score):
            return 0

        query_terms = self.term_tokenizer.terms(
            self.text_contents.contents(input),
        )

        # TODO: Do we really want to use the term set here, not the list?
        #  It seems as if the order of the terms should matter.
        for term in set(query_terms):
            tf1 = self.text_statistics.term_frequency(output1, term)
            tf2 = self.text_statistics.term_frequency(output2, term)
            if isclose(tf1, 0) and tf2 > 0:
                return -1
            if isclose(tf2, 0) and tf1 > 0:
                return 1
        return 0


LB1: Final = lazy_inject(Lb1Axiom, injector)
