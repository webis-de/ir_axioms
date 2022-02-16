from dataclasses import dataclass
from typing import Set

from pandas import DataFrame

from ir_axioms.backend.pyterrier.safe import TransformerBase


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
        assert {"qid", "docno", "label"} == set(qrels.columns)
        return ranking.merge(
            self.qrels,
            on=["qid", "docno"],
            validate="many_to_one"
        )
