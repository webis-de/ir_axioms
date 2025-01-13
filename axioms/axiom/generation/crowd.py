from dataclasses import dataclass
from functools import cached_property
from math import isclose
from pathlib import Path
from typing import TypeAlias, Literal

from pandas import DataFrame, read_json, concat

from axioms.axiom.base import Axiom
from axioms.model import GenerationInput, GenerationOutput
from axioms.axiom.utils import strictly_greater


TrecRagCrowdUtilityType: TypeAlias = Literal[
    "overall",
    "coherence",
    "consistency",
    "correctness",
    "coverage",
]


@dataclass(frozen=True, kw_only=True)
class TrecRagCrowdAxiom(Axiom[GenerationInput, GenerationOutput]):
    ratings_path: Path = Path("data/crowd/ratings.jsonl.gz")
    responses_path: Path = Path("data/crowd/responses.jsonl.gz")

    utility_type: TrecRagCrowdUtilityType
    margin_fraction: float = 0.1

    @cached_property
    def _ratings(self) -> DataFrame:
        df_responses = read_json(self.responses_path, lines=True)
        df_responses["query_hash"] = df_responses.pop("query").map(hash)
        df_responses["answer_text_hash"] = df_responses.pop("raw_text").map(hash)
        df_responses = df_responses[
            ["topic", "query_hash", "response", "answer_text_hash"]
        ]

        df = read_json(
            self.ratings_path,
            lines=True,
        )

        if self.utility_type == "overall":
            df["quality_p_a"] = df["quality_overall_p_a"]
            df["quality_p_b"] = df["quality_overall_p_b"]
        elif self.utility_type == "coherence":
            df["quality_p_a"] = (
                df["coherence_logical_p_a"] + df["coherence_stylistic_p_a"]
            ) / 2
            df["quality_p_b"] = (
                df["coherence_logical_p_b"] + df["coherence_stylistic_p_b"]
            ) / 2
        elif self.utility_type == "consistency":
            df["quality_p_a"] = df["consistency_internal_p_a"]
            df["quality_p_b"] = df["consistency_internal_p_b"]
        elif self.utility_type == "correctness":
            df["quality_p_a"] = df["correctness_topical_p_a"]
            df["quality_p_b"] = df["correctness_topical_p_b"]
        elif self.utility_type == "coverage":
            df["quality_p_a"] = (df["coverage_broad_p_a"] + df["coverage_deep_p_a"]) / 2
            df["quality_p_b"] = (df["coverage_broad_p_b"] + df["coverage_deep_p_b"]) / 2
        else:
            raise ValueError(f"Unknown utility type: {self.utility_type}")

        df = df[
            [
                "query_id",
                "response_a",
                "response_b",
                "quality_p_a",
                "quality_p_b",
            ]
        ]
        df = df.merge(
            right=df_responses[["topic", "query_hash"]].drop_duplicates(),
            left_on="query_id",
            right_on="topic",
        )
        df_responses.drop(columns=["query_hash"], inplace=True)
        df = df.merge(
            right=df_responses,
            left_on=["query_id", "response_a"],
            right_on=["topic", "response"],
        ).rename(columns={"answer_text_hash": "answer_text_hash_a"})
        df = df.merge(
            right=df_responses,
            left_on=["query_id", "response_b"],
            right_on=["topic", "response"],
        ).rename(columns={"answer_text_hash": "answer_text_hash_b"})
        df = df[
            [
                "query_hash",
                "answer_text_hash_a",
                "answer_text_hash_b",
                "quality_p_a",
                "quality_p_b",
            ]
        ]
        return df

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> float:
        df = self._ratings.copy()
        df = df[df["query_hash"] == hash(input.text)]
        hash1 = hash(output1.text)
        hash2 = hash(output2.text)
        if hash1 == hash2:
            return 0
        df1 = df[
            (df["answer_text_hash_a"] == hash1) & (df["answer_text_hash_b"] == hash2)
        ].copy()
        df2 = df[
            (df["answer_text_hash_a"] == hash2) & (df["answer_text_hash_b"] == hash1)
        ].copy()
        df2["quality_p_a"] = 1 - df2["quality_p_a"]
        df2["quality_p_b"] = 1 - df2["quality_p_b"]
        df = concat([df1, df2])
        if len(df) == 0:
            return 0
        df = (
            df.groupby(["answer_text_hash_a", "answer_text_hash_b"])
            .mean()
            .reset_index()
        )
        prob1 = df.iloc[0]["quality_p_a"]
        prob2 = df.iloc[0]["quality_p_b"]
        if isclose(prob1, prob2, rel_tol=self.margin_fraction):
            return 0
        return strictly_greater(prob1, prob2)
