from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True, kw_only=True)
class GenerationInput:
    id: Optional[str] = None
    text: str
    context: Optional[Sequence[str]] = None  # TODO: Would it make more sense to move this to the output?
    reference_output: Optional["GenerationOutput"] = None


@dataclass(frozen=True, kw_only=True)
class GenerationOutput:
    id: Optional[str] = None
    text: str
