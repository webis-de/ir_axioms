from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence

from numpy import ndarray, stack
from sklearn.utils._response import is_classifier
from typing_extensions import Protocol, Self # type: ignore

from axioms.axiom.base import Axiom
from axioms.model import Input, Output, Preference, PreferenceMatrix


class EstimatorAxiom(Axiom[Input, Output], ABC):

    @abstractmethod
    def fit(
        self,
        target: Axiom,
        input: Input,
        outputs: Sequence[Output],
    ) -> None:
        pass




class ScikitLearnEstimator(Protocol):
    def fit(self, x: ndarray, y: ndarray) -> Self:
        pass

    def predict(self, x: ndarray) -> ndarray:
        pass


@dataclass(frozen=True)
class ScikitLearnEstimatorAxiom(EstimatorAxiom, ABC):
    axioms: Sequence[Axiom]
    estimator: ScikitLearnEstimator

    def fit(
        self,
        target: Axiom,
        input: Input,
        outputs: Sequence[Output],
    ) -> None:
        preferences_x = stack([
            axiom.preferences(input, outputs)
            for axiom in self.axioms
        ])
        preferences_x = preferences_x.reshape((-1, len(self.axioms)))

        if is_classifier(self.estimator):
            # If estimator is classifier, normalize target preferences.
            # This will generate the classes: -1, 0, 1
            target = target.normalized()

        preferences_y = target.preferences(input, outputs)
        preferences_y = preferences_y.reshape(-1)

        self.estimator.fit(preferences_x, preferences_y)

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return self.preferences(input, [output1, output2])[0,1]

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        preferences_x = stack([
            axiom.preferences(input, outputs)
            for axiom in self.axioms
        ])
        preferences_x = preferences_x.reshape((-1, len(self.axioms)))

        estimated = self.estimator.predict(preferences_x)
        estimated = estimated.reshape((len(outputs), len(outputs)))

        return estimated