from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    title: str


@dataclass(frozen=True)
class Document:
    id: str


@dataclass(frozen=True)
class TextDocument(Document):
    contents: str


@dataclass(frozen=True)
class RankedDocument(Document):
    score: float
    rank: int


@dataclass(frozen=True)
class RankedTextDocument(TextDocument, RankedDocument):
    pass


@dataclass(frozen=True)
class JudgedRankedDocument(RankedDocument):
    relevance: float


@dataclass(frozen=True)
class JudgedRankedTextDocument(TextDocument, JudgedRankedDocument):
    pass
