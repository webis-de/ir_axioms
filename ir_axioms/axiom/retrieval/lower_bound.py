from dataclasses import dataclass
from math import isclose
from collections.abc import Mapping, Sequence, Set
from typing import Final, Union

from injector import inject, NoInject
from numpy import array, bool_, float_, isclose as numpy_isclose, newaxis, where
from numpy.typing import NDArray

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Query, Document, Preference, PreferenceMatrix
from ir_axioms.tools import TextContents, TermTokenizer, TextStatistics
from ir_axioms.utils.lazy import lazy_inject


def _lb1_preference(
    query_unique_terms: Set[str],
    document1_tf: Mapping[str, float],
    document2_tf: Mapping[str, float],
) -> Preference:
    query_term_only_in_document1 = any(
        document1_tf.get(term, 0) > 0 and document2_tf.get(term, 0) <= 0
        for term in query_unique_terms
    )
    query_term_only_in_document2 = any(
        document2_tf.get(term, 0) > 0 and document1_tf.get(term, 0) <= 0
        for term in query_unique_terms
    )

    if query_term_only_in_document1 and not query_term_only_in_document2:
        return 1
    elif not query_term_only_in_document1 and query_term_only_in_document2:
        return -1
    else:
        return 0


def _require_score(document: Document) -> float:
    if document.score is None:
        raise ValueError("Can only compare scored documents.")
    return document.score


def _similar_scores(
    scores: Sequence[float],
    margin_fraction: float,
) -> NDArray[bool_]:
    """
    Pairwise equivalent of `math.isclose(score1, score2, rel_tol=margin_fraction)`.
    """
    scores_column = array(scores, dtype=float_)[:, newaxis]
    # NumPy's isclose is asymmetric, so check both directions (like math.isclose).
    close_scores = numpy_isclose(
        scores_column,
        scores_column.T,
        rtol=margin_fraction,
        atol=0,
    )
    return close_scores | close_scores.T


@inject
@dataclass(frozen=True, kw_only=True)
class Lb1Axiom(Axiom[Query, Document]):
    text_contents: TextContents[Union[Query, Document]]
    term_tokenizer: TermTokenizer
    text_statistics: TextStatistics[Document]
    margin_fraction: NoInject[float] = 0.1

    def preference(
        self,
        input: Query,
        output1: Document,
        output2: Document,
    ) -> Preference:
        if not isclose(
            _require_score(output1),
            _require_score(output2),
            rel_tol=self.margin_fraction,
        ):
            return 0

        query_terms = self.term_tokenizer.unique_terms(
            self.text_contents.contents(input),
        )

        document1_tf = self.text_statistics.term_frequencies(output1)
        document2_tf = self.text_statistics.term_frequencies(output2)

        return _lb1_preference(
            query_unique_terms=query_terms,
            document1_tf=document1_tf,
            document2_tf=document2_tf,
        )

    def preferences(
        self,
        input: Query,
        outputs: Sequence[Document],
    ) -> PreferenceMatrix:
        # Determine which document pairs have similar scores (documents × documents).
        similar_scores = _similar_scores(
            scores=[_require_score(output) for output in outputs],
            margin_fraction=self.margin_fraction,
        )

        # Get the unique query terms and wrap as list for later deterministic access.
        query_terms = list(
            self.term_tokenizer.unique_terms(
                self.text_contents.contents(input),
            )
        )

        # Compute term frequencies for documents (only with at least one similarly scored document).
        has_similar_scores = similar_scores.sum(axis=1) > 1
        term_frequencies = [
            self.text_statistics.term_frequencies(output) if has_similar_score else {}
            for output, has_similar_score in zip(outputs, has_similar_scores)
        ]

        # Determine which query terms occur in which document (documents × query terms).
        contains_query_term = array(
            [
                document_tf.get(term, 0) > 0
                for document_tf in term_frequencies
                for term in query_terms
            ],
            dtype=bool_,
        ).reshape((len(outputs), len(query_terms)))

        # Find query terms that occur in the first but not the second document (documents × documents).
        # Note: For boolean matrices, the matrix product (documents × query terms) @ (query terms × documents) computes any(contains[i, t] and not contains[j, t] for t in query terms) for every document pair (i, j) at once.
        query_term_only_in_document1 = contains_query_term @ ~contains_query_term.T
        query_term_only_in_document2 = query_term_only_in_document1.T

        # If only the first document has an exclusive query term then return 1, -1 if only the second has one, 0 otherwise or if the scores are not similar.
        return where(
            similar_scores,
            query_term_only_in_document1.astype(float_)
            - query_term_only_in_document2.astype(float_),
            0,
        )


LB1: Final = lazy_inject(Lb1Axiom)
