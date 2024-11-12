from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Tf:
    pass


@dataclass(frozen=True)
class TfIdf:
    pass


@dataclass(frozen=True)
class BM25:
    k_1: float = 1.2
    k_3: float = 8
    b: float = 0.75


@dataclass(frozen=True)
class PL2:
    c: float = 0.1


@dataclass(frozen=True)
class DirichletLM:
    mu: float = 1000


# Aliases
QL = DirichletLM

# Type union
RetrievalModel = Union[Tf, TfIdf, BM25, PL2, DirichletLM]
