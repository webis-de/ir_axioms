from dataclasses import dataclass


@dataclass
class Query:
    id: int
    title: str


@dataclass
class Document:
    id: str
    content: str


@dataclass
class RankedDocument(Document):
    score: float
    rank: int
