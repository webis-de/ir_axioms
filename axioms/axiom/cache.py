from dataclasses import dataclass
from typing import Sequence

from axioms.axiom.base import Axiom
from axioms.model import Input, Output, Preference, PreferenceMatrix


@dataclass(frozen=True, kw_only=True)
class CachedAxiom(Axiom[Input, Output]):
    axiom: Axiom[Input, Output]

    def preference(
        self,
        input: Input,
        output1: Output,
        output2: Output,
    ) -> Preference:
        # TODO: Implement caching.
        raise NotImplementedError()
    
    def preferences(
        self,
        input: Input,
        outputs: Sequence[Output],
    ) -> PreferenceMatrix:
        # TODO: Implement caching.
        raise NotImplementedError()

    def cached(self) -> Axiom[Input, Output]:
        return self
