from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Iterable

from numpy import ndarray, stack
from sklearn.utils._response import is_classifier
from tqdm.auto import tqdm
from typing_extensions import Protocol, Self  # type: ignore

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import Input, Output, Preference, PreferenceMatrix


class EstimatorAxiom(Axiom[Input, Output], ABC):
    @abstractmethod
    def fit(
        self,
        target: Axiom,
        inputs_outputs: Iterable[tuple[Input, Sequence[Output]]],
    ) -> None:
        pass


class ScikitLearnEstimator(Protocol):
    def fit(self, X: ndarray, y: ndarray) -> Self:
        pass

    def predict(self, X: ndarray) -> ndarray:
        pass


@dataclass(frozen=True)
class ScikitLearnEstimatorAxiom(EstimatorAxiom[Input, Output], ABC):
    axioms: Sequence[Axiom]
    estimator: ScikitLearnEstimator

    def fit(
        self,
        target: Axiom,
        inputs_outputs: Iterable[tuple[Input, Sequence[Output]]],
    ) -> None:
        num_axioms = len(self.axioms)

        preferences_x = stack(
            [
                stack(
                    [
                        axiom.preferences(input, outputs)
                        for input, outputs in tqdm(
                            inputs_outputs,
                            desc="Feature preferences",
                            unit="query",
                        )
                    ]
                )
                for axiom in self.axioms
            ]
        )
        print(preferences_x.shape)
        preferences_x = preferences_x.reshape((num_axioms, -1))
        print(preferences_x.shape)

        if is_classifier(self.estimator):
            # If estimator is classifier, normalize target preferences.
            # This will generate the classes: -1, 0, 1
            target = target.normalized()

        preferences_y = stack(
            [
                target.preferences(input, outputs)
                for input, outputs in tqdm(
                    inputs_outputs,
                    desc="Target preferences",
                    unit="query",
                )
            ]
        )
        print(preferences_y.shape)
        preferences_y = preferences_y.reshape(-1)
        print(preferences_y.shape)

        self.estimator.fit(preferences_x, preferences_y)

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return self.preferences(input, [output1, output2])[0, 1]

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        num_axioms = len(self.axioms)
        num_outputs = len(outputs)

        preferences_x = stack(
            [axiom.preferences(input, outputs) for axiom in self.axioms]
        )
        preferences_x = preferences_x.reshape((num_outputs * num_outputs, num_axioms))

        estimated = self.estimator.predict(preferences_x)
        estimated = estimated.reshape((num_outputs, num_outputs))

        return estimated
