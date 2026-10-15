from dataclasses import dataclass
from typing import Optional, Sequence, Type, TYPE_CHECKING
from warnings import warn


@dataclass(frozen=True, kw_only=True)
class GenerationRequest:
    id: Optional[str] = None
    """Unique identifier for the generation input."""
    text: str
    """Request text, e.g., a question or query."""
    context: Optional[Sequence[str]] = None
    """Optional retrieved context (e.g., when evaluating RAG)."""
    reference_output: Optional["GenerationResponse"] = None
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


if TYPE_CHECKING:
    from typing_extensions import deprecated

    @deprecated("GenerationInput is deprecated; use GenerationRequest instead.")
    class GenerationInput(GenerationRequest): ...

    @deprecated("GenerationInput is deprecated; use GenerationRequest instead.")
    class GenerationOutput(GenerationResponse): ...

else:
    from typing import Any

    def __getattr__(name: str) -> Any:
        if name == "OldName":
            warn(
                "GenerationInput is deprecated; use GenerationRequest instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return GenerationRequest
        if name == "GenerationOutput":
            warn(
                "GenerationOutput is deprecated; use GenerationResponse instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return GenerationResponse
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
