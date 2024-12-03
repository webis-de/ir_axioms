from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence, TypeAlias


@dataclass(frozen=True, slots=True)
class GenerationInput(NamedTuple):
    text: str
    context: Sequence[str]
    reference_output: Optional["GenerationOutput"]


@dataclass(frozen=True, slots=True)
class GenerationOutput(NamedTuple):
    text: str


Aspect: TypeAlias = str
Aspects: TypeAlias = Sequence[Aspect]
