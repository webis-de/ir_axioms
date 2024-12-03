from abc import ABC
from dataclasses import dataclass
from typing import Annotated, final

from annotated_types import Interval

from axioms.axiom.base import Axiom
from axioms.model.base import Preference
from axioms.model.generation import GenerationInput, GenerationOutput
from axioms.tools.aspect import AspectExtraction


@dataclass(frozen=True)
class AspectCoverageAxiom(Axiom[GenerationInput, GenerationOutput]):
    aspect_extraction: AspectExtraction
    threshold: Annotated[float, Interval(ge=0, le=1)]

    def preference(
        self,
        input: GenerationInput,
        output1: GenerationOutput,
        output2: GenerationOutput,
    ) -> Preference:
        input_aspects = set(self.aspect_extraction.extract_aspects(text=input.text))
        num_input_aspects = len(input_aspects)
        if num_input_aspects == 0:
            return 0

        # aspects to cover are aspects in context1 minus query aspects

        output1_aspects = (self.aspect_extraction.extract_aspects(text=output1.text),)
        output2_aspects = (self.aspect_extraction.extract_aspects(text=output2.text),)
        covered1 = input_aspects.intersection(output1_aspects)
        covered2 = input_aspects.intersection(output2_aspects)
        coverage1 = len(covered1) / num_input_aspects
        coverage2 = len(covered2) / num_input_aspects
        if coverage1 > self.threshold and coverage2 > self.threshold:
            return 0
        elif coverage1 > self.threshold:
            return 1
        elif coverage2 > self.threshold:
            return -1
        else:
            return 0
