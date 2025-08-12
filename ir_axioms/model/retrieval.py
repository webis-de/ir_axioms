from dataclasses import dataclass
from typing import Type, Any
from warnings import warn


@dataclass(frozen=True)
class Query:
    id: str
    """Unique identifier for the query."""
    text: str | None = None
    """Query text content."""


@dataclass(frozen=True)
class Document:
    id: str
    """Unique identifier for the document."""
    text: str | None = None
    """Document text content."""
    score: float | None = None
    """The document's retrieval score."""
    rank: int | None = None
    """The document's ranking position."""
    relevance: float | None = None
    """The true relevance label of the document."""


# Deprecation aliases for backward compatibility.
def _deprecation_warning(obj: Any, replacement: Type) -> None:
    warn(
        f"{type(obj).__name__} is deprecated. Use {replacement.__name__} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class TextQuery(Query):
    text: str

    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Query)
        super().__init__(*args, **kwargs)
        if self.text is None:
            raise ValueError(f"{type(self).__name__} must have a non-empty text field.")


class TextDocument(Document):
    text: str

    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)
        if self.text is None:
            raise ValueError(f"{type(self).__name__} must have a non-empty text field.")


class ScoredDocument(Document):
    score: float

    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)
        if self.score is None:
            raise ValueError(
                f"{type(self).__name__} must have a non-empty score field."
            )


class RankedDocument(Document):
    rank: int

    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)
        if self.rank is None:
            raise ValueError(f"{type(self).__name__} must have a non-empty rank field.")


class ScoredTextDocument(ScoredDocument, TextDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class RankedTextDocument(RankedDocument, TextDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class RankedScoredDocument(ScoredDocument, RankedDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(RankedScoredDocument, Document)
        super().__init__(*args, **kwargs)


class RankedScoredTextDocument(
    RankedScoredDocument, ScoredTextDocument, RankedTextDocument
):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class JudgedDocument(Document):
    relevance: float

    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)
        if self.relevance is None:
            raise ValueError(
                f"{type(self).__name__} must have a non-empty relevance field."
            )


class JudgedTextDocument(TextDocument, JudgedDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class JudgedScoredDocument(ScoredDocument, JudgedDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class JudgedScoredTextDocument(ScoredTextDocument, JudgedDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class JudgedRankedDocument(RankedDocument, JudgedDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class JudgedRankedTextDocument(RankedTextDocument, JudgedDocument):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class JudgedRankedScoredDocument(
    RankedScoredDocument, JudgedScoredDocument, JudgedRankedDocument
):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)


class JudgedRankedScoredTextDocument(
    JudgedRankedScoredDocument, JudgedScoredTextDocument, JudgedRankedTextDocument
):
    def __init__(self, *args, **kwargs) -> None:
        _deprecation_warning(self, Document)
        super().__init__(*args, **kwargs)
