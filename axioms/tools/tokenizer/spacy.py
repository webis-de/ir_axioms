from dataclasses import dataclass
from functools import cached_property
from typing import Sequence

from spacy import load as spacy_load
from spacy.language import Language

from axioms.tools.tokenizer.base import TermTokenizer, SentenceTokenizer


@dataclass(frozen=True, kw_only=True)
class SpacyTermTokenizer(TermTokenizer):
    language_name: str = "en_core_web_sm"
    lemmatize: bool = False
    lowercase: bool = False

    @cached_property
    def _language(self) -> Language:
        return spacy_load(name=self.language_name)

    def terms(self, text: str) -> Sequence[str]:
        document = self._language(text)
        terms: Sequence[str]
        if self.lemmatize:
            terms = [token.lemma_ for token in document]
        else:
            terms = [token.text for token in document]
        if self.lowercase:
            terms = [term.lower() for term in terms]
        return terms


@dataclass(frozen=True, kw_only=True)
class SpacySentenceTokenizer(SentenceTokenizer):
    language_name: str = "en_core_web_sm"

    @cached_property
    def _language(self) -> Language:
        return spacy_load(name=self.language_name)

    def sentences(self, text: str) -> Sequence[str]:
        document = self._language(text)
        return [sentence.text for sentence in document.sents]
