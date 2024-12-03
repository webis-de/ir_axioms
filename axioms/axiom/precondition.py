from dataclasses import dataclass
from typing import Generic, Sequence

from numpy import zeros_like
from numpy.ma import masked_array

from axioms.axiom.base import Axiom
from axioms.model import (
    Input,
    Output,
    Preference,
    PreferenceMatrix,
)
from axioms.precondition import Precondition


@dataclass(frozen=True, kw_only=True)
class PreconditionAxiom(Axiom[Input, Output], Generic[Input, Output]):
    axiom: Axiom[Input, Output]
    precondition: Precondition[Input, Output]

    def strip_preconditions(self) -> Axiom[Input, Output]:
        axiom = self.axiom
        while isinstance(axiom, PreconditionAxiom):
            axiom = axiom.axiom
        return axiom

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        if not self.precondition.precondition(
            input=input,
            output1=output1,
            output2=output2,
        ):
            return 0
        return self.axiom.preference(
            input=input,
            output1=output1,
            output2=output2,
        )

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        mask = self.precondition.preconditions(
            input=input,
            outputs=outputs,
        )
        if not mask.any():
            return zeros_like(mask)
        # TODO: We could try to isolate the non-masked entries here, and only compute preferences where at least one output of the tuple is not masked.
        return masked_array(
            data=self.axiom.preferences(
                input=input,
                outputs=outputs,
            ),
            mask=mask,
        )
