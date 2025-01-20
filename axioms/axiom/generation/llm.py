from dataclasses import dataclass
from functools import cached_property
from math import isclose, nan, isnan
from pathlib import Path
from typing import Mapping

from injector import NoInject
from pandas import read_json

from axioms.axiom.base import Axiom
from axioms.model import GenerationInput, GenerationOutput
from axioms.axiom.utils import strictly_greater


@dataclass(frozen=True, kw_only=True)
class TrecRagLlmOrigAxiom(Axiom[GenerationInput, GenerationOutput]):
    ratings_path: Path = Path("data/ratings-HuggingFaceTB-SmolLM-360M-Instruct.jsonl")

    margin_fraction: NoInject[float] = 0.0

    @cached_property
    def _ratings(self) -> Mapping[tuple[int, int], float]:
        df = read_json(
            self.ratings_path,
            lines=True,
        )
        return {
            (
                hash(row["query"]),
                hash(row["raw_text"]),
            ): row["probability"]
            for _, row in df.iterrows()
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

        prob1 = self._ratings.get((input_hash, output1_hash), nan)
        prob2 = self._ratings.get((input_hash, output2_hash), nan)
        if isnan(prob1) or isnan(prob2):
            print(f"Missing rating for input: {input.text}")
            print(f"Missing rating for output1: {output1.text}")
            print(f"Missing rating for output2: {output2.text}")

        if self.margin_fraction > 0 and isclose(
            prob1, prob2, rel_tol=self.margin_fraction
        ):
            return 0
        return strictly_greater(prob1, prob2)
