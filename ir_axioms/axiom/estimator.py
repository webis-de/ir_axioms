from abc import abstractmethod, ABC
from dataclasses import dataclass
from itertools import groupby
from typing import Tuple, Iterable, Sequence, Union, Optional, Callable

from numpy import array
from sklearn.base import is_classifier
from sklearn.ensemble import (
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
    GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
    StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.gaussian_process import (
    GaussianProcessClassifier, GaussianProcessRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import (
    BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier,
    ExtraTreeRegressor
)
from tqdm.auto import tqdm

from ir_axioms.axiom.base import Axiom
from ir_axioms.axiom.simple import ORACLE
from ir_axioms.model import RankedDocument, Query, IndexContext, \
    JudgedRankedDocument


class EstimatorAxiom(Axiom, ABC):

    @abstractmethod
    def fit(
            self,
            target: Axiom,
            context: IndexContext,
            query_documents: Iterable[Tuple[Query, RankedDocument]],
            filter_pairs: Optional[Callable[
                [RankedDocument, RankedDocument],
                bool
            ]] = None,
    ) -> None:
        pass

    def fit_oracle(
            self,
            context: IndexContext,
            query_documents: Iterable[Tuple[Query, JudgedRankedDocument]],
            filter_pairs: Optional[Callable[
                [JudgedRankedDocument, JudgedRankedDocument],
                bool
            ]] = None,
    ) -> None:
        return self.fit(ORACLE(), context, query_documents, filter_pairs)


def _query(query_document_pair: Tuple[Query, RankedDocument]) -> Query:
    query, _ = query_document_pair
    return query


def _query_documents_pairs(
        query_document_pairs: Iterable[Tuple[Query, RankedDocument]]
) -> Sequence[Tuple[
    Query,
    RankedDocument,
    RankedDocument,
]]:
    queries = groupby(query_document_pairs, key=_query)
    return [
        (query, document1, document2)
        for query, query_documents in queries
        for _, document1 in query_documents
        for _, document2 in query_documents
    ]


ScikitEstimatorType = Union[
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    GaussianProcessClassifier,
    GaussianProcessRegressor,
    LogisticRegression,
    LinearRegression,
    BernoulliNB,
    CategoricalNB,
    ComplementNB,
    GaussianNB,
    MultinomialNB,
    KNeighborsClassifier,
    KNeighborsRegressor,
    MLPClassifier,
    MLPRegressor,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
]


@dataclass
class ScikitLearnEstimatorAxiom(EstimatorAxiom, ABC):
    axioms: Sequence[Axiom]
    estimator: ScikitEstimatorType
    verbose: bool = False

    def fit(
            self,
            target: Axiom,
            context: IndexContext,
            query_documents: Iterable[Tuple[Query, RankedDocument]],
            filter_pairs: Optional[Callable[
                [RankedDocument, RankedDocument],
                bool
            ]] = None,
    ) -> None:
        axioms = self.axioms

        query_documents_pairs = _query_documents_pairs(query_documents)
        if filter_pairs is not None:
            query_documents_pairs = [
                (query, document1, document2)
                for query, document1, document2 in query_documents_pairs
                if filter_pairs(document1, document2)
            ]

        if self.verbose:
            query_documents_pairs = tqdm(
                query_documents_pairs,
                desc="Collecting axiom preferences",
                unit="document pair",
            )
        preferences_x = array([
            [
                axiom.cached().preference(
                    context,
                    query,
                    document1,
                    document2,
                )
                for axiom in axioms
            ]
            for query, document1, document2 in query_documents_pairs
        ])

        if is_classifier(self.estimator):
            # If estimator is classifier, normalize target preferences.
            # This will generate the classes: -1, 0, 1
            target = target.normalized()

        preferences_y = array([
            target.preference(
                context,
                query,
                document1,
                document2,
            )
            for query, document1, document2 in query_documents_pairs
        ])

        self.estimator.fit(preferences_x, preferences_y)

    def preference(
            self,
            context: IndexContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preferences_x = array([
            [
                axiom.cached().preference(
                    context,
                    query,
                    document1,
                    document2,
                )
                for axiom in self.axioms
            ]
        ])
        estimated = self.estimator.predict(preferences_x)
        return float(estimated[0])
