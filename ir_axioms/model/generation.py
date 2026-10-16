from dataclasses import dataclass
from typing import Optional, Sequence

from typing_extensions import deprecated


@dataclass(frozen=True, kw_only=True)
class GenerationRequest:
    id: str | None = None
    """Unique identifier for the generation input."""
    text: str
    """Request text, e.g., a question or query."""
    context: Sequence[str] | None = None
    """Optional retrieved context (e.g., when evaluating RAG)."""
    reference_output: Optional["GenerationResponse"] = None
    """Optional ground-truth refeerence response."""


@deprecated("Use GenerationRequest instead.")
class GenerationInput(GenerationRequest): ...


@dataclass(frozen=True, kw_only=True)
class GenerationResponse:
    id: str | None = None
    """Unique identifier for the generation output."""
    text: str
    """Text response generated."""
    context: Sequence[str] | None = None
    """Override context from input if necessary."""


@deprecated("Use GenerationRequest instead.")
class GenerationOutput(GenerationResponse): ...
