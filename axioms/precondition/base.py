from typing import Generic, Sequence, Protocol

from numpy import array

from axioms.model.base import (
    Input,
    Mask,
    MaskMatrix,
    Output,
)


class Precondition(Generic[Input, Output], Protocol):
    def precondition(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Mask:
        # TODO: Documentation.
        pass

    def preconditions(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> MaskMatrix:
        # TODO: Documentation.
        return array(
            [
                [
                    self.precondition(
                        input=input,
                        output1=output1,
                        output2=output2,
                    )
                    for output2 in outputs
                ]
                for output1 in outputs
            ]
        )
