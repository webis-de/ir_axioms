from dataclasses import dataclass
from math import isclose
from typing import AbstractSet, Final, Mapping, Sequence, Union

from injector import inject
from numpy import array, float_
from tqdm.auto import tqdm

from ir_axioms.axiom.base import Axiom
from ir_axioms.dependency_injection import injector
from ir_axioms.axiom.utils import strictly_fewer, strictly_greater
from ir_axioms.model import PreferenceMatrix, Query, Document, Preference
from ir_axioms.tools import TextContents, TermTokenizer, TextStatistics
from ir_axioms.utils.lazy import lazy_inject


@inject
@dataclass(frozen=True, kw_only=True)
class Lnc1Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]
    margin_fraction: float = 0.1

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
            isclose(
                self.text_statistics.term_frequency(output1, term),
                self.text_statistics.term_frequency(output2, term),
                rel_tol=self.margin_fraction,
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

    def _preference(
        self,
        query_unique_terms: AbstractSet[str],
        document1_terms: Sequence[str],
        document2_terms: Sequence[str],
        document1_term_frequencies: Mapping[str, float],
        document2_term_frequencies: Mapping[str, float],
    ) -> Preference:
        sum_document1 = 0
        sum_document2 = 0

        for query_term in query_unique_terms:
            tf_d1 = document1_term_frequencies[query_term]
            tf_d2 = document2_term_frequencies[query_term]

            len_d1 = sum(1 for term in document1_terms if term != query_term)
            len_d2 = sum(1 for term in document2_terms if term != query_term)

            if len_d1 == len_d2:
                if tf_d1 > tf_d2:
                    sum_document1 += 1
                elif tf_d2 > tf_d1:
                    sum_document2 += 1

        return strictly_greater(sum_document1, sum_document2)

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
        document1_term_frequencies = {
            query_term: self.text_statistics.term_frequency(output1, query_term)
            for query_term in query_unique_terms
        }
        document2_term_frequencies = {
            query_term: self.text_statistics.term_frequency(output2, query_term)
            for query_term in query_unique_terms
        }
        return self._preference(
            query_unique_terms=query_unique_terms,
            document1_terms=document1_terms,
            document2_terms=document2_terms,
            document1_term_frequencies=document1_term_frequencies,
            document2_term_frequencies=document2_term_frequencies,
        )

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document_terms = [
            self.term_tokenizer.terms(
                self.text_contents.contents(output),
            )
            for output in tqdm(
                outputs,
                total=len(outputs),
                desc="Tokenize",
                unit="document",
            )
        ]
        document_term_frequencies = [
            {
                query_term: self.text_statistics.term_frequency(output, query_term)
                for query_term in query_unique_terms
            }
            for output in tqdm(
                outputs,
                total=len(outputs),
                desc="Term frequencies",
                unit="document",
            )
        ]
        return array(
            [
                self._preference(
                    query_unique_terms=query_unique_terms,
                    document1_terms=document1_terms,
                    document2_terms=document2_terms,
                    document1_term_frequencies=document1_term_frequencies,
                    document2_term_frequencies=document2_term_frequencies,
                )
                for document1_terms, document1_term_frequencies in zip(
                    document_terms, document_term_frequencies
                )
                for document2_terms, document2_term_frequencies in zip(
                    document_terms, document_term_frequencies
                )
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


TF_LNC: Final = lazy_inject(TfLncAxiom, injector)
