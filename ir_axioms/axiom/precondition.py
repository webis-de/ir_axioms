from abc import ABC
from dataclasses import dataclass
from typing import Sequence

from numpy import where, zeros_like

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import (
    Input,
    Output,
    Preference,
    PreferenceMatrix,
)
from ir_axioms.precondition.base import Precondition


@dataclass(frozen=True, kw_only=True)
class PreconditionMixin(Axiom[Input, Output], ABC):
    precondition: Precondition[Input, Output]

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
        return super().preference(  # type: ignore[safe-super]
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
            return zeros_like(mask).astype(float)
        # ADDITION: We could try to isolate the non-masked entries here, and only compute preferences where at least one output of the tuple is not masked.
        preferences = super().preferences(
            input=input,
            outputs=outputs,
        )
        # Only keep preferences where the precondition holds.
        return where(mask, preferences, 0)
