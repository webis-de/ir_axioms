from dataclasses import dataclass


@dataclass(frozen=True, unsafe_hash=True)
class Query:
    title: str


@dataclass(frozen=True, unsafe_hash=True)
class Document:
    id: str


@dataclass(frozen=True, unsafe_hash=True)
class TextDocument(Document):
    contents: str


@dataclass(frozen=True, unsafe_hash=True)
class RankedDocument(Document):
    score: float
    rank: int


@dataclass(frozen=True, unsafe_hash=True)
class RankedTextDocument(TextDocument, RankedDocument):
    pass