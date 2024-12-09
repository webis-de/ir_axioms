from dataclasses import dataclass
from typing import Optional, Sequence, TypeAlias


@dataclass(frozen=True, kw_only=True)
class GenerationInput:
    id: Optional[str] = None
    text: str
    context: Optional[Sequence[str]] = None
    reference_output: Optional["GenerationOutput"] = None


@dataclass(frozen=True, kw_only=True)
class GenerationOutput:
    id: Optional[str] = None
    text: str


Aspect: TypeAlias = str
Aspects: TypeAlias = Sequence[Aspect]
