from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Sequence, Iterable

from numpy import ndarray, stack, concatenate, array, expand_dims
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
    def fit(self, X: ndarray, y: ndarray) -> Self: ...
    def predict(self, X: ndarray) -> ndarray: ...


@dataclass(frozen=True)
class ScikitLearnEstimatorAxiom(EstimatorAxiom[Input, Output], ABC):
    axioms: Sequence[Axiom[Input, Output]]
    estimator: ScikitLearnEstimator

    def fit(
        self,
        target: Axiom[Input, Output],
        inputs_outputs: Iterable[tuple[Input, Sequence[Output]]],
    ) -> None:
        # Compute preferences of every axiom for the inputs and outputs.
        # The individual preference matices for each common input will be flattened, so that, after stacking, the final array contains columns for each axiom and rows for each combination of input and two outputs.
        # Shape: |input+output1+output2 combinations| x |axioms|
        preferences_x = stack(
            [
                concatenate(
                    [
                        axiom.preferences(
                            input=input,
                            outputs=outputs,
                        ).reshape(-1)
                        for input, outputs in tqdm(
                            inputs_outputs,
                            desc="Feature preferences",
                            unit="query",
                        )
                    ]
                )
                for axiom in self.axioms
            ],
            axis=-1,
        )

        # If the estimator is a classifier, the targets need to be discretized.
        # By normalizing the target preferences, we ensure that the target values are one of -1, 0, and 1, i.e., the classes for the classifier.
        if is_classifier(self.estimator):
            target = target.normalized()

        # Compute the target preferences for the inputs and outputs.
        # The individual preference matrices for each common input will be flattened, so that, after concatenating, the final array contains the target preferences for each combination of input and two outputs.
        # Shape: |input+output1+output2 combinations|
        preferences_y = concatenate(
            [
                target.preferences(input, outputs).reshape(-1)
                for input, outputs in tqdm(
                    inputs_outputs,
                    desc="Target preferences",
                    unit="query",
                )
            ]
        )

        # With the flattened inputs and flattened targets, now fit the estimator.
        self.estimator.fit(preferences_x, preferences_y)

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        # Compute the preferences of each axiom.
        # Shape: |axioms|
        preferences_x = array(
            [
                axiom.preference(
                    input=input,
                    output1=output1,
                    output2=output2,
                )
                for axiom in self.axioms
            ]
        )
        # Reshape to ensure it is a 2D array with one row.
        # Shape: 1 x |axioms|
        preferences_x = expand_dims(preferences_x, axis=0)

        # Predict the preference using the estimator.
        # Shape: 1
        estimated = self.estimator.predict(preferences_x)

        # Return the estimated preference as a scalar.
        return estimated[0]

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        # Compute the preferences of each axiom for the inputs and outputs.
        # The individual preference matrices for each common input will be flattened, so that, after stacking, the final array contains columns for each axiom and rows for each combination of input and two outputs.
        # Shape: (|outputs|*|outputs|) x |axioms|
        preferences_x = stack(
            [
                axiom.preferences(
                    input=input,
                    outputs=outputs,
                ).reshape(-1)
                for axiom in self.axioms
            ],
            axis=-1,
        )

        # Predict the preferences using the estimator.
        # This will aggregate along the last axis, i.e., the axioms.
        # Shape: (|outputs|*|outputs|)
        estimated = self.estimator.predict(preferences_x)

        # Reshape the estimated preferences to ensure it is a 2D preference matrix.
        # Shape: |outputs| x |outputs|
        estimated = estimated.reshape((len(outputs), -1))

        # Return the estimated preferences as a preference matrix.
        return estimated
