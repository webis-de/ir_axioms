from dataclasses import dataclass
from typing import Set, Optional, Sequence, Callable

from pandas import DataFrame, Series

from ir_axioms.backend.pyterrier import ContentsAccessor
from ir_axioms.backend.pyterrier.safe import TransformerBase
from ir_axioms.model import (
    RankedDocument, RankedTextDocument, JudgedRankedTextDocument,
    JudgedRankedDocument, Query
)


def require_columns(
        ranking: DataFrame,
        expected_columns: Set[str],
) -> None:
    columns: Set[str] = set(ranking.columns)
    missing_columns: Set[str] = expected_columns - columns
    if len(missing_columns) > 0:
        raise ValueError(
            f"Expected columns "
            f"{', '.join(expected_columns)} but got columns "
            f"{', '.join(columns)} (missing columns "
            f"{', '.join(missing_columns)})."
        )


def load_documents(
        ranking: DataFrame,
        contents_accessor: Optional[ContentsAccessor] = "text",
) -> Sequence[RankedDocument]:
    require_columns(ranking, {"docno", "rank", "score"})

    has_contents_accessor = (
            contents_accessor is not None and
            isinstance(contents_accessor, str) and
            contents_accessor in ranking.columns
    )
    parser: Callable[[Series], RankedDocument]

    if "label" in ranking.columns:
        if has_contents_accessor:
            def parser(row: Series) -> RankedDocument:
                return JudgedRankedTextDocument(
                    id=str(row["docno"]),
                    contents=str(row[contents_accessor]),
                    score=float(row["score"]),
                    rank=int(row["rank"]),
                    relevance=float(row["label"]),
                )
        else:
            def parser(row: Series) -> RankedDocument:
                return JudgedRankedDocument(
                    id=str(row["docno"]),
                    score=float(row["score"]),
                    rank=int(row["rank"]),
                    relevance=float(row["label"]),
                )
    else:
        if has_contents_accessor:
            def parser(row: Series) -> RankedDocument:
                return RankedTextDocument(
                    id=str(row["docno"]),
                    contents=str(row[contents_accessor]),
                    score=float(row["score"]),
                    rank=int(row["rank"]),
                )
        else:
            def parser(row: Series) -> RankedDocument:
                return RankedDocument(
                    id=str(row["docno"]),
                    score=float(row["score"]),
                    rank=int(row["rank"]),
                )
    return [
        parser(row)
        for _, row in ranking.iterrows()
    ]


def load_queries(ranking: DataFrame) -> Sequence[Query]:
    require_columns(ranking, {"query"})
    return [
        Query(row["query"])
        for _, row in ranking.iterrows()
    ]


@dataclass(frozen=True)
class FilterTopicsTransformer(TransformerBase):
    """
    Retain only queries that are contained in the topics.
    """

    topics: DataFrame

    def transform(self, ranking: DataFrame) -> DataFrame:
        return ranking[ranking["qid"].isin(self.topics["qid"])]


@dataclass(frozen=True)
class FilterQrelsTransformer(TransformerBase):
    """
    Retain only query-document pairs that are contained in the qrels.
    """

    qrels: DataFrame

    def transform(self, ranking: DataFrame) -> DataFrame:
        return ranking[
            ranking["qid"].isin(self.qrels["qid"]) &
            ranking["docno"].isin(self.qrels["docno"])
            ]


@dataclass(frozen=True)
class JoinQrelsTransformer(TransformerBase):
    """
    Join query-document pairs with their relevance labels.
    """

    qrels: DataFrame

    def transform(self, ranking: DataFrame) -> DataFrame:
        qrels = self.qrels
        require_columns(qrels, {"qid", "docno", "label"})
        return ranking.merge(
            self.qrels,
            on=["qid", "docno"],
            how="left"
        )
