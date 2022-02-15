from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Tuple, Iterable, Sequence, Union

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

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import RankedDocument, Query, IndexContext


class EstimatorAxiom(Axiom, ABC):

    @abstractmethod
    def fit(
            self,
            target: Axiom,
            context: IndexContext,
            query_documents: Iterable[Tuple[Query, Sequence[RankedDocument]]],
    ) -> None:
        pass


def _query_document_pairs(
        query_rankings: Iterable[Tuple[Query, Sequence[RankedDocument]]]
) -> Iterable[Tuple[
    Query,
    RankedDocument,
    RankedDocument,
]]:
    return (
        (query, document1, document2)
        for query, ranking in query_rankings
        for document1 in ranking
        for document2 in ranking
    )


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
    estimator: ScikitEstimatorType = RandomForestClassifier

    def fit(
            self,
            target: Axiom,
            context: IndexContext,
            query_rankings: Iterable[Tuple[Query, Sequence[RankedDocument]]]
    ) -> None:
        query_document_pairs = _query_document_pairs(query_rankings)
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
            for query, document1, document2 in query_document_pairs
        ])

        if is_classifier(self.estimator):
            # If estimator is classifier, normalize target preferences.
            # This will generate the classes: -1, 0, 1
            target = target.normalized()
        target = target.cached()

        preferences_y = array([
            target.preference(
                context,
                query,
                document1,
                document2,
            )
            for query, document1, document2 in query_document_pairs
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
