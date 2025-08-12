from dataclasses import dataclass
from typing import Optional, Sequence, Type
from warnings import warn


@dataclass(frozen=True, kw_only=True)
class GenerationRequest:
    id: Optional[str] = None
    """Unique identifier for the generation input."""
    text: str
    """Request text, e.g., a question or query."""
    context: Optional[Sequence[str]] = None
    """Optional retrieved context (e.g., when evaluating RAG)."""
    reference_output: Optional["GenerationOutput"] = None
    """Optional ground-truth refeerence response."""


@dataclass(frozen=True, kw_only=True)
class GenerationResponse:
    id: Optional[str] = None
    """Unique identifier for the generation output."""
    text: str
    """Text response generated."""
    context: Optional[Sequence[str]] = None
    """Override context from input if necessary."""


# Deprecation aliases for backward compatibility.
def _deprecation_warning(cls: Type, replacement: Type) -> None:
    warn(
        f"{cls.__name__} is deprecated. Use {replacement.__name__} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class GenerationInput(GenerationRequest):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(GenerationInput, GenerationRequest)
        super().__init__(*args, **kwargs)


class GenerationOutput(GenerationResponse):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(GenerationOutput, GenerationResponse)
        super().__init__(*args, **kwargs)
