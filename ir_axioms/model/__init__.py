from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    title: str


@dataclass(frozen=True)
class Document:
    id: str


@dataclass(frozen=True)
class RankedDocument(Document):
    score: float
    rank: int
