from abc import abstractmethod, ABC
from dataclasses import dataclass
from math import log
from typing import Union, FrozenSet, Sequence


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
class JudgedDocument(Document):
    relevance: float
