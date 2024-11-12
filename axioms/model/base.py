from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Query:
    title: str


@dataclass(frozen=True, kw_only=True)
class Document:
    id: str


@dataclass(frozen=True, kw_only=True)
class TextDocument(Document):
    contents: str


@dataclass(frozen=True, kw_only=True)
class RankedDocument(Document):
    score: float
    rank: int


@dataclass(frozen=True, kw_only=True)
class RankedTextDocument(TextDocument, RankedDocument):
    pass


@dataclass(frozen=True, kw_only=True)
class JudgedRankedDocument(RankedDocument):
    relevance: float


@dataclass(frozen=True, kw_only=True)
class JudgedRankedTextDocument(TextDocument, JudgedRankedDocument):
    pass
