from collections.abc import Iterator, Iterable
from dataclasses import dataclass
from functools import cached_property
from math import inf
from statistics import StatisticsError
from typing import AbstractSet
from warnings import warn

from spacy import load as spacy_load
from spacy.language import Language
from spacy.tokens import Doc
from textacy.extract.keyterms.yake import yake

from ir_axioms.tools.aspects.base import AspectExtraction


@dataclass(frozen=True)
class YakeAspectExtraction(AspectExtraction):
    """
    Extract key phrases using YAKE (via textacy) as aspects.

    For texts with fewer than two distinct content words (e.g., single-word
    queries), YAKE cannot score candidates. Then, the (lemmatized) unigram
    candidates are returned without scoring.
    """

    language_name: str = "en_core_web_sm"
    ngram_range: tuple[int, int] = (1, 3)
    include_pos: Iterable[str] = ("NOUN", "PROPN", "ADJ")
    top_n: int = 10
    threshold: float = inf

    @cached_property
    def _language(self) -> Language:
        # YAKE needs part-of-speech tags, lemmas, and sentence boundaries.
        return spacy_load(name=self.language_name, exclude=["ner"])

    def _extract(self, document: Doc) -> AbstractSet[str]:
        try:
            keyphrases = yake(
                doc=document,
                ngrams=list(range(self.ngram_range[0], self.ngram_range[1] + 1)),
                include_pos=set(self.include_pos),
                topn=self.top_n,
            )
        except StatisticsError:
            # YAKE normalizes word frequencies by the mean and standard deviation
            # over all distinct non-stopword tokens, which is undefined if there
            # are fewer than two of them (e.g., for single-word queries like
            # "cat"). With at most one candidate there is nothing to rank,
            # so we fall back to returning the unigram candidates YAKE would
            # have scored. As there is no score, the threshold is not applied.
            warn(
                "YAKE cannot score texts with fewer than two distinct content "
                "words. Falling back to unscored unigram candidates.",
                UserWarning,
                stacklevel=3,
            )
            return self._extract_fallback(document)
        return {phrase for phrase, score in keyphrases if score <= self.threshold}

    def _extract_fallback(self, document: Doc) -> AbstractSet[str]:
        if self.ngram_range[0] > 1 or self.top_n < 1:
            return set()
        # Same candidate selection and normalization as YAKE's unigrams.
        return {
            token.lemma_
            for token in document
            if not (token.is_stop or token.is_punct or token.is_space)
            and token.pos_ in self.include_pos
        }

    def aspects(self, text: str) -> AbstractSet[str]:
        return self._extract(self._language(text))

    def iter_aspects(self, texts: Iterable[str]) -> Iterator[AbstractSet[str]]:
        for document in self._language.pipe(texts):
            yield self._extract(document)
