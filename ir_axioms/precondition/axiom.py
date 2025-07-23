from dataclasses import dataclass
from typing import Generic, Literal, Sequence, TYPE_CHECKING

from numpy import sign

from ir_axioms.model import (
    Input,
    Output,
    Mask,
    MaskMatrix,
)
from ir_axioms.precondition.base import Precondition

if TYPE_CHECKING:
    from ir_axioms.axiom.base import Axiom


@dataclass(frozen=True, kw_only=True)
class AxiomPrecondition(Precondition[Input, Output], Generic[Input, Output]):
    axiom: "Axiom[Input, Output]"
    expected_sign: Literal[1, 0, -1] = 0
    strip_preconditions: bool = True

    def precondition(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Mask:
        preference = self.axiom.preference(input, output1, output2)
        return sign(preference) == self.expected_sign

    def preconditions(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> MaskMatrix:
        preferences = self.axiom.preferences(input, outputs)
        return sign(preferences) == self.expected_sign
