from dataclasses import dataclass
from typing import Generic, Optional, Sequence

from joblib import Parallel, delayed
from numpy import array

from axioms.axiom.base import Axiom
from axioms.model import (
    Input,
    Output,
    Preference,
    PreferenceMatrix,
)



@dataclass(frozen=True, kw_only=True)
class ParllelAxiom(Axiom[Input, Output], Generic[Input, Output]):
    axiom: Axiom[Input, Output]
    n_jobs: Optional[int] = None

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        return self.axiom.preference(input, output1, output2)

    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        


        @delayed
        def _preference(
            output1: Output,
            output2: Output,
        ) -> float:
            return self.preference(
                input=input,
                output1=output1,
                output2=output2,
            )

        with Parallel(n_jobs=self.n_jobs) as parallel:
            preferences: Sequence[float] = parallel(
                _preference(output1, output2)
                for output1 in outputs
                for output2 in outputs
            )

        return array(preferences).reshape(
            len(outputs),
            len(outputs)
        )
