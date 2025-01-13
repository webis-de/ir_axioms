from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

from keybert import KeyBERT, KeyLLM
from keybert.llm import BaseLLM

from axioms.tools.aspects.base import AspectExtraction


@dataclass(frozen=True)
class KeyBertAspectExtraction(AspectExtraction):
    model: str = "all-MiniLM-L6-v2"
    # TODO: Make extraction configurable.

    @cached_property
    def _model(self) -> KeyBERT:
        return KeyBERT(model=self.model)

    def aspects(self, input: str) -> Sequence[str]:
        keywords = self._model.extract_keywords(input)
        return [keyword for keyword, _ in keywords]


@dataclass(frozen=True)
class KeyLlmAspectExtraction(AspectExtraction):
    llm: BaseLLM
    # TODO: Make extraction configurable.

    @cached_property
    def _model(self) -> KeyLLM:
        return KeyLLM(llm=self.llm)

    def aspects(self, input: str) -> Sequence[str]:
        keywords = self._model.extract_keywords(input)
        return [keyword for keyword, _ in keywords]
