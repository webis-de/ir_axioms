from dataclasses import dataclass
from functools import cached_property
from typing import AbstractSet, Iterator, Iterable

from spacy import load as spacy_load
from spacy.language import Language
from textacy.extract.keyterms import yake

from ir_axioms.tools.aspects.base import AspectExtraction


@dataclass(frozen=True)
class YakeAspectExtraction(AspectExtraction):
    language_name: str = "en_core_web_sm"
    ngram_range: tuple[int, int] = (1, 3)
    top_n: int = 10
    threshold: float = 1.0

    @cached_property
    def _language(self) -> Language:
        return spacy_load(name=self.language_name, enable=["parser"])

    def aspects(self, input: str) -> AbstractSet[str]:
        document = self._language(input)
        keyphrases = yake(
            doc=document,
            ngrams=list(range(*self.ngram_range)),
            topn=self.top_n,
        )
        return {phrase for phrase, score in keyphrases if score <= self.threshold}

    def iter_aspects(self, texts: Iterable[str]) -> Iterator[AbstractSet[str]]:
        documents = self._language.pipe(texts)
        keyphrases = (
            yake(
                doc=document,
                ngrams=list(range(*self.ngram_range)),
                topn=self.top_n,
            )
            for document in documents
        )
        for keys in keyphrases:
            yield {phrase for phrase, score in keys if score <= self.threshold}
