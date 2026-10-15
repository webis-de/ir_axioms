from dataclasses import dataclass
from warnings import catch_warnings, simplefilter

from typing_extensions import deprecated


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


@deprecated("Use Query instead.")
class TextQuery(Query):
    text: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.text is None:
            raise ValueError(f"{type(self).__name__} must have a non-empty text field.")


@deprecated("Use Document instead.")
class TextDocument(Document):
    text: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.text is None:
            raise ValueError(f"{type(self).__name__} must have a non-empty text field.")


@deprecated("Use Document instead.")
class ScoredDocument(Document):
    score: float

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.score is None:
            raise ValueError(
                f"{type(self).__name__} must have a non-empty score field."
            )


@deprecated("Use Document instead.")
class RankedDocument(Document):
    rank: int

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.rank is None:
            raise ValueError(f"{type(self).__name__} must have a non-empty rank field.")


# Subclassing deprecated classes emits deprecation warnings at class definition,
# i.e., when importing this module. Only warn when the classes are actually used.
with catch_warnings():
    simplefilter("ignore", category=DeprecationWarning)

    @deprecated("Use Document instead.")
    class ScoredTextDocument(ScoredDocument, TextDocument): ...

    @deprecated("Use Document instead.")
    class RankedTextDocument(RankedDocument, TextDocument): ...

    @deprecated("Use Document instead.")
    class RankedScoredDocument(ScoredDocument, RankedDocument): ...

    @deprecated("Use Document instead.")
    class RankedScoredTextDocument(
        RankedScoredDocument, ScoredTextDocument, RankedTextDocument
    ): ...

    @deprecated("Use Document instead.")
    class JudgedDocument(Document):
        relevance: float

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if self.relevance is None:
                raise ValueError(
                    f"{type(self).__name__} must have a non-empty relevance field."
                )

    @deprecated("Use Document instead.")
    class JudgedTextDocument(TextDocument, JudgedDocument): ...

    @deprecated("Use Document instead.")
    class JudgedScoredDocument(ScoredDocument, JudgedDocument): ...

    @deprecated("Use Document instead.")
    class JudgedScoredTextDocument(ScoredTextDocument, JudgedDocument): ...

    @deprecated("Use Document instead.")
    class JudgedRankedDocument(RankedDocument, JudgedDocument): ...

    @deprecated("Use Document instead.")
    class JudgedRankedTextDocument(RankedTextDocument, JudgedDocument): ...

    @deprecated("Use Document instead.")
    class JudgedRankedScoredDocument(
        RankedScoredDocument, JudgedScoredDocument, JudgedRankedDocument
    ): ...

    @deprecated("Use Document instead.")
    class JudgedRankedScoredTextDocument(
        JudgedRankedScoredDocument, JudgedScoredTextDocument, JudgedRankedTextDocument
    ): ...
