from typing import Generic, Sequence, Protocol

from numpy import array
from tqdm.auto import tqdm

from ir_axioms.model.base import (
    Input,
    Mask,
    MaskMatrix,
    Output,
)


class Precondition(Generic[Input, Output], Protocol):
    """
    A precondition can be used to restrict an axiom's preference computation to only those outputs that meet a certain condition (e.g., similar length).
    """

    def precondition(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Mask:
        """
        Check if the precondition holds for a pair of outputs given an input.

        :param input: Common input for both outputs.
        :param output1: One output for the common input.
        :param output2: Another output for the common input.
        :return: ``True`` if a preference can be computed for the outputs, ``False`` otherwise.
        """
        ...

    def preconditions(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> MaskMatrix:
        """
        Check if the precondition holds for all pairs of outputs given an input.

        :param input: Common input for all outputs.
        :param outputs: The outputs for the common input.
        :return: A mask matrix, where the ij-th entry corresponds to whether the precondition holds for the i-th and j-th output.
        """
        return array(
            list(
                tqdm(
                    (
                        self.precondition(
                            input=input,
                            output1=output1,
                            output2=output2,
                        )
                        for output1 in outputs
                        for output2 in outputs
                    ),
                    desc="Preconditions",
                    total=len(outputs) * len(outputs),
                )
            )
        ).reshape((len(outputs), len(outputs)))
