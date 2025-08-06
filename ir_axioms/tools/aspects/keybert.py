from ir_axioms.utils.libraries import is_keybert_installed

if is_keybert_installed():
    from dataclasses import dataclass
    from functools import cached_property
    from typing import AbstractSet, Iterator, Iterable

    from keybert import KeyBERT

    from ir_axioms.tools.aspects.base import AspectExtraction

    @dataclass(frozen=True)
    class KeyBertAspectExtraction(AspectExtraction):
        model: str = "all-MiniLM-L6-v2"
        ngram_range: tuple[int, int] = (1, 3)
        top_n: int = 10
        threshold: float = 0.0

        @cached_property
        def _model(self) -> KeyBERT:
            return KeyBERT(model=self.model)

        def aspects(self, text: str) -> AbstractSet[str]:
            keyphrases: list[tuple[str, float]] = self._model.extract_keywords(
                docs=text,
                keyphrase_ngram_range=self.ngram_range,
                top_n=self.top_n,
            )  # type: ignore
            return {phrase for phrase, score in keyphrases if score >= self.threshold}

        def iter_aspects(self, texts: Iterable[str]) -> Iterator[AbstractSet[str]]:
            keyphrases: list[list[tuple[str, float]]] = self._model.extract_keywords(
                docs=list(texts),
                keyphrase_ngram_range=self.ngram_range,
                top_n=self.top_n,
            )  # type: ignore
            for keys in keyphrases:
                yield {phrase for phrase, score in keys if score >= self.threshold}
else:
    KeyBertAspectExtraction = NotImplemented  # type: ignore
