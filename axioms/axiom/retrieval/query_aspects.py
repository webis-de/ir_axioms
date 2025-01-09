from dataclasses import dataclass, field
from typing import Final, Sequence, Set, AbstractSet, List, Union

from injector import inject, NoInject
from numpy import array, float_
from tqdm.auto import tqdm

from axioms.axiom.base import Axiom
from axioms.axiom.precondition import PreconditionMixin
from axioms.dependency_injection import injector
from axioms.precondition.base import Precondition
from axioms.precondition.length import LEN
from axioms.axiom.utils import strictly_greater, approximately_equal
from axioms.model import PreferenceMatrix, Query, Document, Preference
from axioms.tools import (
    TermSimilarity,
    TextContents,
    TermTokenizer,
    IndexStatistics,
    TextStatistics,
)
from axioms.utils.lazy import lazy_inject


def _vocabulary_overlap(
    vocabulary1: AbstractSet[str],
    vocabulary2: AbstractSet[str],
):
    """
    Vocabulary overlap as calculated by the Jaccard coefficient.
    """
    intersection_length = len(vocabulary1 & vocabulary2)
    if intersection_length == 0:
        return 0
    return intersection_length / (
        len(vocabulary1) + len(vocabulary2) - intersection_length
    )


@inject
@dataclass(frozen=True, kw_only=True)
class RegAxiom(PreconditionMixin[Query, Document], Axiom[Query, Document]):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]
    term_similarity: TermSimilarity
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input)
        )

        min_average_similarity_terms = (
            self.term_similarity.min_average_similarity_terms(query_unique_terms)
        )
        if len(min_average_similarity_terms) == 0:
            return 0

        if all(
            self.text_statistics.term_frequency(output1, term)
            > self.text_statistics.term_frequency(output2, term)
            for term in min_average_similarity_terms
        ):
            return 1
        elif all(
            self.text_statistics.term_frequency(output1, term)
            < self.text_statistics.term_frequency(output2, term)
            for term in min_average_similarity_terms
        ):
            return -1
        else:
            return 0


REG: Final = lazy_inject(RegAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AntiRegAxiom(PreconditionMixin[Query, Document], Axiom[Query, Document]):
    """
    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.

    Modified to use maximum similarity instead of minimum similarity.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]
    term_similarity: TermSimilarity
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )

        max_average_similarity_terms = (
            self.term_similarity.max_average_similarity_terms(query_unique_terms)
        )
        if len(max_average_similarity_terms) == 0:
            return 0

        if all(
            self.text_statistics.term_frequency(output1, term)
            > self.text_statistics.term_frequency(output2, term)
            for term in max_average_similarity_terms
        ):
            return 1
        elif all(
            self.text_statistics.term_frequency(output1, term)
            < self.text_statistics.term_frequency(output2, term)
            for term in max_average_similarity_terms
        ):
            return -1
        else:
            return 0


ANTI_REG: Final = lazy_inject(AntiRegAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AspectRegAxiom(Axiom[Query, Document]):
    """
    Similar to REG but follows the query aspect clustering
    from the paper and then counts the number of aspects covered
    in each document.

    Reference:
    Zheng, W., Fang, H.: Query aspect based term weighting regularization
    in information retrieval. In: Gurrin, C., et al. (eds.) ECIR 2010.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    index_statistics: IndexStatistics
    term_similarity: TermSimilarity
    term_discriminator_margin_fraction: float = 0.1

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_terms = self.term_tokenizer.terms(
            self.text_contents.contents(input),
        )
        query_unique_terms = set(query_terms)
        document1_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output1),
        )
        document1_unique_terms = set(document1_terms)
        document2_terms = self.term_tokenizer.terms(
            self.text_contents.contents(output2),
        )
        document2_unique_terms = set(document2_terms)

        if len(query_unique_terms) == 0:
            return 0

        term_discriminators = {
            self.index_statistics.inverse_document_frequency(term)
            for term in query_unique_terms
        }
        if not approximately_equal(
            *term_discriminators,
            margin_fraction=self.term_discriminator_margin_fraction,
        ):
            # Require same term discriminator for all query terms.
            return 0

        average_similarity = self.term_similarity.average_similarity(
            query_unique_terms, query_unique_terms
        )

        query_aspects: List[Set[str]] = [{term} for term in query_unique_terms]

        # Iterate aspect 1 from start.
        for i1 in range(0, len(query_aspects) - 1, +1):
            a1 = query_aspects[i1]
            # Iterate aspect 2 from end.
            for i2 in range(len(query_aspects) - 1, i1 + 1, -1):
                a2 = query_aspects[i2]

                # Is any term pair similar enough to merge the aspects?
                if any(
                    self.term_similarity.similarity(term1, term2) > average_similarity
                    for term1 in a1
                    for term2 in a2
                ):
                    # Merge aspect 2 into aspect 1.
                    a1.update(a2)
                    # Remove merged aspect 2.
                    query_aspects.pop(i2)

        count_document1_aspects = {
            1
            for aspect in query_aspects
            if not document1_unique_terms.isdisjoint(aspect)
        }
        count_document2_aspects = {
            1
            for aspect in query_aspects
            if not document2_unique_terms.isdisjoint(aspect)
        }
        return strictly_greater(
            len(count_document1_aspects) > 0,
            len(count_document2_aspects) > 0,
        )


ASPECT_REG: Final = lazy_inject(AspectRegAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class AndAxiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output1),
        )
        document2_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output2),
        )

        all_query_terms1 = query_unique_terms <= document1_unique_terms
        all_query_terms2 = query_unique_terms <= document2_unique_terms
        return strictly_greater(all_query_terms1, all_query_terms2)

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document_unique_terms = (
            self.term_tokenizer.unique_terms(
                self.text_contents.contents(output),
            )
            for output in outputs
        )
        all_query_terms = [
            query_unique_terms <= terms
            for terms in tqdm(
                document_unique_terms,
                total=len(outputs),
                desc="Query term overlap",
                unit="document",
            )
        ]
        return array(
            [
                strictly_greater(all_query_terms1, all_query_terms2)
                for all_query_terms1 in all_query_terms
                for all_query_terms2 in all_query_terms
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


AND: Final = lazy_inject(AndAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LenAndAxiom(PreconditionMixin[Query, Document], AndAxiom):
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)


LEN_AND: Final = lazy_inject(LenAndAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class ModifiedAndAxiom(Axiom[Query, Document]):
    """
    Modified AND:
    One document contains a larger subset of query terms.
    """

    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output1),
        )
        document2_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output2),
        )

        query_terms1 = query_unique_terms & document1_unique_terms
        query_terms2 = query_unique_terms & document2_unique_terms
        return strictly_greater(len(query_terms1), len(query_terms2))

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document_unique_terms = (
            self.term_tokenizer.unique_terms(
                self.text_contents.contents(output),
            )
            for output in outputs
        )
        num_query_terms = [
            len(query_unique_terms & terms)
            for terms in tqdm(
                document_unique_terms,
                total=len(outputs),
                desc="Query term overlap",
                unit="document",
            )
        ]
        return array(
            [
                strictly_greater(num_query_terms1, num_query_terms2)
                for num_query_terms1 in num_query_terms
                for num_query_terms2 in num_query_terms
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


M_AND: Final = lazy_inject(ModifiedAndAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LenModifiedAndAxiom(PreconditionMixin[Query, Document], ModifiedAndAxiom):
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)


LEN_M_AND: Final = lazy_inject(LenModifiedAndAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class DivAxiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document1_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output1),
        )
        document2_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(output2),
        )

        overlap1 = _vocabulary_overlap(
            vocabulary1=query_unique_terms,
            vocabulary2=document1_unique_terms,
        )
        overlap2 = _vocabulary_overlap(
            vocabulary1=query_unique_terms,
            vocabulary2=document2_unique_terms,
        )

        return strictly_greater(overlap2, overlap1)

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        query_unique_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )
        document_unique_terms = (
            self.term_tokenizer.unique_terms(
                self.text_contents.contents(output),
            )
            for output in outputs
        )
        overlaps = [
            _vocabulary_overlap(
                vocabulary1=query_unique_terms,
                vocabulary2=terms,
            )
            for terms in tqdm(
                document_unique_terms,
                total=len(outputs),
                desc="Vocabulary overlap",
                unit="document",
            )
        ]
        return array(
            [
                strictly_greater(overlap2, overlap1)
                for overlap1 in overlaps
                for overlap2 in overlaps
            ],
            dtype=float_,
        ).reshape((len(outputs), len(outputs)))


DIV: Final = lazy_inject(DivAxiom, injector)


@inject
@dataclass(frozen=True, kw_only=True)
class LenDivAxiom(PreconditionMixin[Query, Document], DivAxiom):
    precondition: NoInject[Precondition[Query, Document]] = field(default_factory=LEN)


LEN_DIV: Final = lazy_inject(LenDivAxiom, injector)
