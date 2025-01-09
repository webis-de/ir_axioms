from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Sequence

from spacy import load as spacy_load
from spacy.language import Language

from axioms.tools.aspects.base import AspectExtraction


@dataclass(frozen=True)
class SpacyNounChunksAspectExtraction(AspectExtraction):
    language_name: str = "en_core_web_sm"
    chunks: Literal["direct", "root", "both"] = "direct"

    @cached_property
    def _language(self) -> Language:
        return spacy_load(name=self.language_name, enable=["parser"])

    def aspects(self, input: str) -> Sequence[str]:
        document = self._language(input)
        chunks = []
        if self.chunks == "direct" or self.chunks == "both":
            chunks += [chunk.text.lower() for chunk in document.noun_chunks]
        if self.chunks == "root" or self.chunks == "both":
            chunks += [chunk.root.text.lower() for chunk in document.noun_chunks]
        return chunks
