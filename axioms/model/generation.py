from dataclasses import dataclass
from typing import Optional, Sequence, TypeAlias


@dataclass(frozen=True)
class GenerationInput:
    text: str
    context: Optional[Sequence[str]] = None
    reference_output: Optional["GenerationOutput"] = None


@dataclass(frozen=True)
class GenerationOutput:
    text: str


Aspect: TypeAlias = str
Aspects: TypeAlias = Sequence[Aspect]
