from dataclasses import dataclass
from typing import Hashable, Sequence, TypeVar

from axioms.axiom.base import Axiom
from axioms.model import Preference, PreferenceMatrix


HashableInput = TypeVar("HashableInput", bound=Hashable)
HashableOutput = TypeVar("HashableOutput", bound=Hashable)


@dataclass(frozen=True, kw_only=True)
class CachedAxiom(Axiom[HashableInput, HashableOutput]):
    axiom: Axiom[HashableInput, HashableOutput]

    def preference(
        self,
        input: HashableInput,
        output1: HashableOutput,
        output2: HashableOutput,
    ) -> Preference:
        # TODO: Implement caching.
        raise NotImplementedError()

    def preferences(
        self,
        input: HashableInput,
        outputs: Sequence[HashableOutput],
    ) -> PreferenceMatrix:
        # TODO: Implement caching.
        raise NotImplementedError()

    def cached(self) -> Axiom[HashableInput, HashableOutput]:
        return self
