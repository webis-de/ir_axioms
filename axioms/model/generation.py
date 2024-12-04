from dataclasses import dataclass
from typing import Optional, Sequence, TypeAlias


@dataclass(frozen=True)
class GenerationInput:
    text: str
    context: Sequence[str]
    reference_output: Optional["GenerationOutput"]


@dataclass(frozen=True)
class GenerationOutput:
    text: str


Aspect: TypeAlias = str
Aspects: TypeAlias = Sequence[Aspect]
