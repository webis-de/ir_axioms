from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    id: str


@dataclass(frozen=True)
class TextQuery(Query):
    text: str


@dataclass(frozen=True)
class Document:
    id: str


@dataclass(frozen=True)
class TextDocument(Document):
    text: str


@dataclass(frozen=True)
class ScoredDocument(Document):
    score: float


@dataclass(frozen=True)
class RankedDocument(Document):
    rank: int


@dataclass(frozen=True)
class ScoredTextDocument(ScoredDocument, TextDocument):
    pass


@dataclass(frozen=True)
class RankedTextDocument(RankedDocument, TextDocument):
    pass


@dataclass(frozen=True)
class RankedScoredDocument(ScoredDocument, RankedDocument):
    pass


@dataclass(frozen=True)
class RankedScoredTextDocument(
    RankedScoredDocument, ScoredTextDocument, RankedTextDocument
):
    pass


@dataclass(frozen=True)
class JudgedDocument(Document):
    relevance: float


@dataclass(frozen=True)
class JudgedScoredDocument(ScoredDocument, JudgedDocument):
    pass


@dataclass(frozen=True)
class JudgedScoredTextDocument(ScoredTextDocument, JudgedDocument):
    pass


@dataclass(frozen=True)
class JudgedRankedDocument(RankedDocument, JudgedDocument):
    pass


@dataclass(frozen=True)
class JudgedRankedTextDocument(RankedTextDocument, JudgedDocument):
    pass


@dataclass(frozen=True)
class JudgedRankedScoredDocument(
    RankedScoredDocument, JudgedScoredDocument, JudgedRankedDocument
):
    pass


@dataclass(frozen=True)
class JudgedRankedScoredTextDocument(
    JudgedRankedScoredDocument, JudgedScoredTextDocument, JudgedRankedTextDocument
):
    pass


# TODO: Consolidate classes as `Protocol`s and only expose dataclasses for some.
