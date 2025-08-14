from dataclasses import dataclass
from functools import cached_property
from math import isclose
from pathlib import Path
from typing import Literal

from pandas import DataFrame, read_json, concat, Series
from typing_extensions import TypeAlias  # type: ignore

from ir_axioms.axiom.base import Axiom
from ir_axioms.model import GenerationInput, GenerationOutput
from ir_axioms.axiom.utils import strictly_greater

TrecRagNuggetScoreType: TypeAlias = Literal[
    "all",
    "vital",
    "weighted",
]


@dataclass(frozen=True, kw_only=True)
class TrecRagNuggetAxiom(Axiom[GenerationInput, GenerationOutput]):
    assignments_path: Path = Path("data/nugget_assignment.20241108.jl")
    score_type: TrecRagNuggetScoreType = "all"
    strict: bool = False
    margin_fraction: float = 0.0

    @cached_property
    def _assignments(self) -> DataFrame:
        df = read_json(
            self.assignments_path,
            lines=True,
        )
        df["query_hash"] = df.pop("query").map(hash)
        df["answer_text_hash"] = df.pop("answer_text").map(hash)
        df = df.drop(
            columns=[
                "qid",
                "run_id",
                "response_length",
            ]
        )
        df = df.explode(column="nuggets")
        df = concat(
            [df, df.pop("nuggets").apply(Series)],
            axis=1,
        )
        df = df.drop(
            columns=[
                "text",
            ]
        )
        df["assignment_score"] = (
            df["assignment"]
            .map(
                {
                    "support": 1.0,
                    "partial_support": 0.0 if self.strict else 0.5,
                }
            )
            .fillna(0.0)
        )
        return df

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> float:
        df = self._assignments
        df = df[df["query_hash"] == hash(input.text)]
        df1 = df[df["answer_text_hash"] == hash(output1.text)]
        df2 = df[df["answer_text_hash"] == hash(output2.text)]
        if len(df1) == 0 or len(df2) == 0:
            return 0

        score_type: TrecRagNuggetScoreType = self.score_type
        if score_type == "all":
            score1 = df1["assignment_score"].sum() / len(df1)
            score2 = df2["assignment_score"].sum() / len(df2)
        elif score_type == "vital":
            df1 = df1[df1["importance"] == "vital"]
            df2 = df2[df2["importance"] == "vital"]
            score1 = df1["assignment_score"].sum() / len(df1)
            score2 = df2["assignment_score"].sum() / len(df2)
        elif score_type == "weighted":
            df1_vital = df1[df1["importance"] == "vital"]
            df2_vital = df2[df2["importance"] == "vital"]
            df1_okay = df1[df1["importance"] != "vital"]
            df2_okay = df2[df2["importance"] != "vital"]
            score1 = (
                df1_vital["assignment_score"].sum()
                + 0.5 * df1_okay["assignment_score"].sum()
            ) / (len(df1_vital) + 0.5 * len(df1_okay))
            score2 = (
                df2_vital["assignment_score"].sum()
                + 0.5 * df2_okay["assignment_score"].sum()
            ) / (len(df2_vital) + 0.5 * len(df2_okay))
        else:
            raise ValueError(f"Unknown score type: {score_type}")

        if isclose(
            score1,
            score2,
            rel_tol=self.margin_fraction,
        ):
            return 0

        return strictly_greater(score1, score2)
