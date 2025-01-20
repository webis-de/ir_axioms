from dataclasses import dataclass
from functools import cached_property
from math import isclose, nan
from pathlib import Path
from typing import TypeAlias, Literal, Mapping

from pandas import read_json

from axioms.axiom.base import Axiom
from axioms.model import GenerationInput, GenerationOutput
from axioms.axiom.utils import strictly_greater


TrecRagCrowdUtilityType: TypeAlias = Literal[
    "overall",
    "logical coherence",
    "stylistic coherence",
    "coherence",
    "internal consistency",
    "consistency",
    "topical correctness",
    "correctness",
    "broad coverage",
    "deep coverage",
    "coverage",
]


@dataclass(frozen=True, kw_only=True)
class TrecRagCrowdAxiom(Axiom[GenerationInput, GenerationOutput]):
    ratings_path: Path = Path("data/crowd/ratings.jsonl.gz")
    responses_path: Path = Path("data/crowd/responses.jsonl.gz")

    utility_type: TrecRagCrowdUtilityType
    margin_fraction: float = 0.1

    @cached_property
    def _ratings(self) -> Mapping[tuple[int, int, int], tuple[float, float]]:
        df = read_json(self.ratings_path, lines=True)
        df_responses = read_json(self.responses_path, lines=True)

        # Merge in the query text.
        df = df.merge(
            right=df_responses[["topic", "query"]].drop_duplicates(),
            left_on="query_id",
            right_on="topic",
        )

        # Merge in the response text.
        df = df.merge(
            right=df_responses[["topic", "response", "raw_text"]],
            left_on=["query_id", "response_a"],
            right_on=["topic", "response"],
        ).rename(columns={"raw_text": "raw_text_a"})
        df = df.merge(
            right=df_responses[["topic", "response", "raw_text"]],
            left_on=["query_id", "response_b"],
            right_on=["topic", "response"],
        ).rename(columns={"raw_text": "raw_text_b"})

        # Select the quality columns based on the utility type.
        if self.utility_type == "overall":
            df["utility_p_a"] = df["quality_overall_p_a"]
            df["utility_p_b"] = df["quality_overall_p_b"]
        elif self.utility_type == "logical coherence":
            df["utility_p_a"] = df["coherence_logical_p_a"]
            df["utility_p_b"] = df["coherence_logical_p_b"]
        elif self.utility_type == "stylistic coherence":
            df["utility_p_a"] = df["coherence_stylistic_p_a"]
            df["utility_p_b"] = df["coherence_stylistic_p_b"]
        elif self.utility_type == "coherence":
            df["utility_p_a"] = (
                df["coherence_stylistic_p_a"] + df["coherence_logical_p_a"]
            ) / 2
            df["utility_p_b"] = (
                df["coherence_stylistic_p_b"] + df["coherence_logical_p_b"]
            ) / 2
        elif (
            self.utility_type == "internal consistency"
            or self.utility_type == "consistency"
        ):
            df["utility_p_a"] = df["consistency_internal_p_a"]
            df["utility_p_b"] = df["consistency_internal_p_b"]
        elif (
            self.utility_type == "topical correctness"
            or self.utility_type == "correctness"
        ):
            df["utility_p_a"] = df["correctness_topical_p_a"]
            df["utility_p_b"] = df["correctness_topical_p_b"]
        elif self.utility_type == "broad coverage":
            df["utility_p_a"] = df["coverage_broad_p_a"]
            df["utility_p_b"] = df["coverage_broad_p_b"]
        elif self.utility_type == "deep coverage":
            df["utility_p_a"] = df["coverage_deep_p_a"]
            df["utility_p_b"] = df["coverage_deep_p_b"]
        elif self.utility_type == "coverage":
            df["utility_p_a"] = (df["coverage_broad_p_a"] + df["coverage_deep_p_a"]) / 2
            df["utility_p_b"] = (df["coverage_broad_p_b"] + df["coverage_deep_p_b"]) / 2
        else:
            raise ValueError(f"Unknown utility type: {self.utility_type}")

        return {
            (
                hash(row["query"]),
                hash(row[f"raw_text_{suffix1}"]),
                hash(row[f"raw_text_{suffix2}"]),
            ): (
                row[f"utility_p_{suffix1}"],
                row[f"utility_p_{suffix2}"],
            )
            for _, row in df.iterrows()
            for suffix1, suffix2 in (
                ("a", "b"),
                ("b", "a"),
            )
        }

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> float:
        input_hash = hash(input.text)
        output1_hash = hash(output1.text)
        output2_hash = hash(output2.text)

        if hash(output1.text) == hash(output2.text):
            return 0

        prob1, prob2 = self._ratings.get(
            (input_hash, output1_hash, output2_hash), (nan, nan)
        )

        if self.margin_fraction > 0 and isclose(
            prob1, prob2, rel_tol=self.margin_fraction
        ):
            return 0
        return strictly_greater(prob1, prob2)
